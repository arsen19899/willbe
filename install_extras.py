"""Доустановка недостающих компонентов CallQA.

Ставит то, что не входит в базовую установку:

  ffmpeg      — чтение mp3 / m4a / ogg. Кладётся в bin/ рядом с программой:
                права администратора и правка PATH не нужны.
  CUDA-libs   — cuBLAS и cuDNN для работы распознавания на видеокарте.
                torch для этого НЕ нужен: faster-whisper работает на
                CTranslate2. Тип вычислений подбирается по видеокарте —
                на Pascal (GTX 10xx) float16 не поддерживается.
  локальная   — подключение Ollama / LM Studio вместо платного API:
  модель        смысловые пункты считаются бесплатно и без интернета.
  llm_key     — платный ключ API, если локальной модели нет.
  pyannote    — точное разделение голосов в моно-записях (нужен torch).
  hf_token    — токен HuggingFace, без него pyannote не скачает модель.

  python install_extras.py           спросить по каждому пункту
  python install_extras.py --all     поставить всё, что можно, без вопросов
  python install_extras.py --check   только показать, чего не хватает
  python install_extras.py --torch   только torch под видеокарту (GTX 10 -> 2.8.0+cu126)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "bin"
LOG = ROOT / "extras_log.txt"
IS_WIN = os.name == "nt"

FFMPEG_URLS = [
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl.zip",
]

# Версия драйвера NVIDIA -> подходящий набор колёс torch
CUDA_CHANNELS = [
    (580.0, "cu130"),
    (570.0, "cu128"),
    (560.0, "cu126"),
    (550.0, "cu124"),
    (525.0, "cu121"),
    (450.0, "cu118"),
]

# Старые видеокарты. Свежие сборки torch их больше не поддерживают: с 2.8
# cu128+ собираются без Pascal, CUDA 13 не знает Pascal вовсе. Последняя
# проверенная сборка с Pascal (GTX 10xx, способность 6.1) — 2.8.0+cu126.
# pyannote 4.x требует torch >= 2.8, так что она ему подходит.
PASCAL_TORCH = {
    "tag": "cu126",
    "min_driver": 560.0,
    "packages": ["torch==2.8.0", "torchaudio==2.8.0"],
    "extra": ["torchcodec==0.7.*"],
}


# --------------------------------------------------------------------------- #
def say(msg: str = "") -> None:
    print(msg, flush=True)


def log(text: str) -> None:
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        f.write(text.rstrip() + "\n")


def run(cmd: List[str], timeout: int = 5400,
        env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    log(f"\n$ {' '.join(str(c) for c in cmd)}")
    try:
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           env=env)
    except FileNotFoundError:
        log("  -> команда не найдена")
        return 127, ""
    except subprocess.TimeoutExpired:
        log("  -> превышено время ожидания")
        return 124, ""
    out = (p.stdout or "") + (p.stderr or "")
    log(out)
    return p.returncode, out


def ask(question: str, default_yes: bool = True, assume: Optional[bool] = None) -> bool:
    if assume is not None:
        return assume
    hint = "[Д/н]" if default_yes else "[д/Н]"
    try:
        answer = input(f"  {question} {hint}: ").strip().lower()
    except EOFError:
        return default_yes
    if not answer:
        return default_yes
    return answer[0] in ("д", "y", "1")


def ask_secret(prompt: str) -> str:
    try:
        import getpass

        return getpass.getpass(f"  {prompt}: ").strip()
    except Exception:
        try:
            return input(f"  {prompt}: ").strip()
        except EOFError:
            return ""


def have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def pip(*args: str) -> int:
    code, _ = run([sys.executable, "-m", "pip", "install", *args])
    return code


# --------------------------------------------------------------------------- #
#  Видеокарта
# --------------------------------------------------------------------------- #
def detect_gpu() -> Dict[str, object]:
    """Определяет видеокарту NVIDIA без установленного torch."""
    info: Dict[str, object] = {"present": False, "name": "", "driver": 0.0}

    smi = shutil.which("nvidia-smi")
    if not smi and IS_WIN:
        cand = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "nvidia-smi.exe"
        if cand.is_file():
            smi = str(cand)
    if smi:
        code, out = run([smi, "--query-gpu=name,driver_version",
                         "--format=csv,noheader"], timeout=60)
        if code == 0 and out.strip():
            first = out.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if parts:
                info["present"] = True
                info["name"] = parts[0]
                if len(parts) > 1:
                    m = re.match(r"(\d+)\.(\d+)", parts[1])
                    if m:
                        info["driver"] = float(f"{m.group(1)}.{m.group(2)}")
            return info

    if IS_WIN:  # драйвера может не быть в PATH — спросим систему
        code, out = run(["powershell", "-NoProfile", "-Command",
                         "(Get-CimInstance Win32_VideoController).Name"], timeout=60)
        if code == 0 and "nvidia" in out.lower():
            for line in out.splitlines():
                if "nvidia" in line.lower():
                    info["present"] = True
                    info["name"] = line.strip()
                    break
    return info


def cuda_channel(driver: float) -> Optional[str]:
    for need, tag in CUDA_CHANNELS:
        if driver >= need:
            return tag
    return None


# --------------------------------------------------------------------------- #
#  ffmpeg
# --------------------------------------------------------------------------- #
def ffmpeg_present() -> Optional[str]:
    for name in ("ffmpeg.exe", "ffmpeg"):
        p = BIN / name
        if p.is_file():
            return str(p)
    return shutil.which("ffmpeg")


def download(url: str, dest: Path) -> bool:
    say(f"    качаю {url.split('/')[-1]} ...")
    log(f"\n# download {url}")
    try:
        import requests

        with requests.get(url, stream=True, timeout=60,
                          headers={"User-Agent": "CallQA-installer"}) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 18):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    _progress(done, total)
    except Exception as exc:
        log(f"requests не смог: {exc}")
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as exc2:
            log(f"urllib не смог: {exc2}")
            say(f"    не удалось скачать: {exc2}")
            return False
    sys.stdout.write("\r" + " " * 60 + "\r")
    return dest.is_file() and dest.stat().st_size > 100_000


def _progress(done: int, total: int) -> None:
    mb = done / 1e6
    if total:
        pct = int(100 * done / total)
        bar = int(pct / 4)
        sys.stdout.write(f"\r    [{'█' * bar}{'·' * (25 - bar)}] {pct:3d}%  {mb:.0f} МБ")
    else:
        sys.stdout.write(f"\r    скачано {mb:.0f} МБ")
    sys.stdout.flush()


def extract_ffmpeg(archive: Path, dest: Path) -> set:
    """Достаёт ffmpeg.exe и ffprobe.exe из архива, где бы внутри они ни лежали.

    Имена участников архива не используются как пути на диске — берётся
    только базовое имя, поэтому zip-slip невозможен.
    """
    wanted = {"ffmpeg.exe", "ffprobe.exe", "ffmpeg", "ffprobe"}
    extracted = set()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if member.endswith("/"):
                continue
            base = Path(member).name.lower()
            if base not in wanted:
                continue
            target = dest / base
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            try:
                target.chmod(0o755)
            except OSError:
                pass
            extracted.add(base)
    return extracted


def install_ffmpeg(assume: Optional[bool] = None) -> bool:
    found = ffmpeg_present()
    if found:
        say(f"  ffmpeg уже есть: {found}")
        return True

    if not IS_WIN:
        say("  ffmpeg: установите пакетом системы, например  sudo apt install ffmpeg")
        return False

    say("  ffmpeg — нужен для чтения mp3, m4a, ogg (около 80 МБ).")
    if not ask("Скачать и положить в bin/ рядом с программой?", True, assume):
        return False

    BIN.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "ffmpeg.zip"
        ok = False
        for url in FFMPEG_URLS:
            if download(url, archive):
                ok = True
                break
        if not ok:
            say("    [!] Скачать не удалось. Проверьте интернет или поставьте вручную:")
            say("        https://www.gyan.dev/ffmpeg/builds/")
            return False

        say("    распаковываю ...")
        try:
            extracted = extract_ffmpeg(archive, BIN)
        except zipfile.BadZipFile:
            say("    [!] Архив повреждён.")
            return False

    if "ffmpeg.exe" not in extracted:
        say("    [!] В архиве не нашёлся ffmpeg.exe")
        return False

    exe = BIN / "ffmpeg.exe"
    code, out = run([str(exe), "-version"], timeout=60)
    if code != 0:
        say("    [!] ffmpeg распакован, но не запускается — см. extras_log.txt")
        return False
    ver = out.splitlines()[0] if out else ""
    say(f"    готово: {ver[:60]}")
    say(f"    лежит в {BIN}")
    return True


# --------------------------------------------------------------------------- #
#  CUDA-библиотеки для faster-whisper
#
#  Важно: faster-whisper работает на CTranslate2 и НЕ требует torch.
#  Ему нужны cuBLAS и cuDNN. На Windows их нет в pip, поэтому берём
#  готовый архив Purfview и кладём в bin/ — он уже добавляется в PATH.
# --------------------------------------------------------------------------- #
CUDA_LIBS_API = ("https://api.github.com/repos/Purfview/"
                 "whisper-standalone-win/releases/tags/libs")


def cuda_runtime_ready() -> bool:
    """Можно ли реально считать на видеокарте.

    Раньше здесь спрашивалось число CUDA-устройств — и этого оказалось
    мало. Драйвер показывает устройство даже когда cuBLAS и cuDNN не
    установлены: считать на нём ещё нечем. Установщик из-за этого решал,
    что всё готово, и библиотеки не ставил, а распознавание падало на
    первом же файле с «Library cublas64_12.dll is not found».
    """
    probe = (
        "import sys;sys.path.insert(0, r'{root}');"
        "from callqa.asr import cuda_device_present, cuda_libs_loadable;"
        "print('DEV', cuda_device_present());"
        "print('LIBS', cuda_libs_loadable())"
    ).format(root=str(ROOT))
    code, out = run([sys.executable, "-c", probe], timeout=180)
    state = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("DEV", "LIBS"):
            state[parts[0]] = parts[1] == "True"
    return bool(state.get("DEV") and state.get("LIBS"))


def pick_cuda_asset(assets: List[dict]) -> Optional[dict]:
    """Выбирает архив под CUDA 12 / cuDNN 9 для нужной системы."""
    want_win = IS_WIN

    def score(a: dict) -> int:
        n = (a.get("name") or "").lower()
        s = 0
        if want_win and "linux" in n:
            return -100
        if not want_win and "linux" not in n:
            return -100
        if "cuda12" in n.replace(".", "").replace("_", "") or "cuda_12" in n:
            s += 50
        if "v12" in n or "_12" in n:
            s += 20
        if "cudnn9" in n.replace(".", "").replace("_", "") or "__v9" in n:
            s += 20
        if "cuda11" in n.replace(".", "").replace("_", ""):
            s -= 40
        return s

    ranked = sorted(assets, key=score, reverse=True)
    return ranked[0] if ranked and score(ranked[0]) > -100 else None


def extract_archive(archive: Path, dest: Path) -> int:
    """Распаковывает zip или 7z, раскладывая файлы плоско в dest."""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    count = 0
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if member.endswith("/"):
                    continue
                base = Path(member).name
                with z.open(member) as src, open(dest / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
        return count
    if name.endswith(".7z"):
        try:
            import py7zr
        except ImportError:
            say("    ставлю py7zr для распаковки .7z ...")
            if pip("py7zr") != 0:
                raise RuntimeError("не удалось поставить py7zr")
            import py7zr
        with tempfile.TemporaryDirectory() as tmp:
            with py7zr.SevenZipFile(archive, "r") as z:
                z.extractall(tmp)
            for p in Path(tmp).rglob("*"):
                if p.is_file():
                    shutil.copy2(p, dest / p.name)
                    count += 1
        return count
    raise RuntimeError(f"неизвестный формат архива: {archive.name}")


def install_cuda_libs(gpu: Dict[str, object], assume: Optional[bool] = None) -> bool:
    if not gpu["present"]:
        say("  Видеокарта NVIDIA не найдена — распознавание пойдёт на процессоре.")
        return False
    if cuda_runtime_ready():
        say("  CUDA для распознавания уже работает")
        return True

    say(f"  Видеокарта: {gpu['name']}")
    say("  Драйвер её показывает, но считать нечем: нет cuBLAS/cuDNN.")
    say("  Для работы на видеокарте нужны библиотеки cuBLAS и cuDNN (около 300 МБ).")
    say("  Ускорение распознавания — примерно в 10 раз. torch для этого НЕ нужен.")
    if not ask("Скачать и положить в bin/?", True, assume):
        return False

    try:
        import requests

        r = requests.get(CUDA_LIBS_API, timeout=60,
                         headers={"User-Agent": "CallQA-installer"})
        r.raise_for_status()
        assets = r.json().get("assets") or []
    except Exception as exc:
        say(f"    [!] Не удалось получить список файлов: {exc}")
        say("        Скачайте вручную и распакуйте в папку bin/:")
        say("        https://github.com/Purfview/whisper-standalone-win/releases/tag/libs")
        return False

    asset = pick_cuda_asset(assets)
    if not asset:
        say("    [!] Подходящий архив не найден. Скачайте вручную в bin/:")
        say("        https://github.com/Purfview/whisper-standalone-win/releases/tag/libs")
        return False

    say(f"    выбран: {asset['name']} ({asset.get('size', 0) / 1e6:.0f} МБ)")
    BIN.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / asset["name"]
        if not download(asset["browser_download_url"], archive):
            return False
        say("    распаковываю ...")
        try:
            n = extract_archive(archive, BIN)
        except Exception as exc:
            say(f"    [!] Распаковать не удалось: {exc}")
            return False
    say(f"    распаковано файлов: {n} в {BIN}")

    os.environ["PATH"] = str(BIN) + os.pathsep + os.environ.get("PATH", "")
    adder = getattr(os, "add_dll_directory", None)   # Windows: PATH для DLL не работает
    if adder is not None:
        try:
            adder(str(BIN))
        except OSError:
            pass
    if cuda_runtime_ready():
        say("    CUDA работает")
        return True
    say("    [!] Библиотеки на месте, но CTranslate2 видеокарту пока не видит.")
    say("        Часто помогает перезапуск программы. Проверьте: run_cli.bat doctor")
    return True


def configure_gpu_settings(gpu: Dict[str, object]) -> None:
    """Прописывает в настройки device=cuda и правильный тип вычислений.

    На Pascal (GTX 10xx) float16 не поддерживается — нужен int8_float32,
    иначе CTranslate2 молча откатится и потеряет скорость.
    """
    if not gpu["present"]:
        return
    try:
        sys.path.insert(0, str(ROOT))
        from callqa.asr import compute_capability
        from callqa.config import DEFAULT_CONFIG_PATH, Config
    except Exception:
        return

    cc = compute_capability()
    if cc is None:
        return
    ctype = "float16" if cc >= 7.0 else ("int8_float32" if cc >= 6.1 else "float32")

    current = os.path.join(os.path.dirname(DEFAULT_CONFIG_PATH), "current.yaml")
    path = current if os.path.exists(current) else DEFAULT_CONFIG_PATH
    cfg = Config.load(path)
    cfg.asr.device = "cuda"
    cfg.asr.compute_type = ctype
    cfg.save(current)
    say(f"    в настройках: device=cuda, compute_type={ctype} "
        f"(вычислительная способность {cc})")
    if cc < 7.0:
        say("    float16 на вашей видеокарте не поддерживается — это нормально,")
        say("    int8_float32 на ней быстрее.")


# --------------------------------------------------------------------------- #
#  Локальная языковая модель
# --------------------------------------------------------------------------- #
OLLAMA_MODELS = [
    ("qwen3.5:9b", "6.6 ГБ", "лучшее на русском для 8 ГБ; есть в I:\\localmodel"),
    ("qwen3:8b", "5.2 ГБ", "прежняя модель, чуть меньше"),
    ("qwen3:4b", "2.6 ГБ", "вдвое быстрее, качество чуть ниже"),
    ("llama3.1:8b", "4.7 ГБ", "альтернатива, хорошо держит диалог"),
    ("gemma3:4b", "3.3 ГБ", "самая быстрая из пригодных"),
]


def _ollama_cfg():
    """Рабочий конфиг с адресом локальной Ollama — для поиска и запуска ollama.exe."""
    sys.path.insert(0, str(ROOT))
    from callqa.config import DEFAULT_CONFIG_PATH, Config

    current = os.path.join(os.path.dirname(DEFAULT_CONFIG_PATH), "current.yaml")
    cfg = Config.load(current if os.path.exists(current) else DEFAULT_CONFIG_PATH)
    cfg.llm.enabled = True
    cfg.llm.provider = "openai"
    if ":11434" not in (cfg.llm.base_url or ""):
        cfg.llm.base_url = "http://127.0.0.1:11434/v1"
    return cfg


def ollama_cmd(*args: str) -> Tuple[Optional[List[str]], Optional[Dict[str, str]]]:
    """ollama.exe (в том числе портативная I:\\localmodel) и окружение для него."""
    try:
        from callqa import ollama_local

        cmd, env = ollama_local.cli(_ollama_cfg(), *args)
        return cmd, (env if cmd else None)
    except Exception:
        w = shutil.which("ollama")
        return ([w, *args] if w else None), None


def start_ollama() -> Tuple[bool, str]:
    try:
        from callqa import ollama_local

        return ollama_local.ensure_running(_ollama_cfg(), log=lambda m: say(f"    {m}"))
    except Exception as exc:
        return False, str(exc)


def find_local_llm() -> List[dict]:
    try:
        sys.path.insert(0, str(ROOT))
        from callqa.llm import discover_local_servers

        return discover_local_servers()
    except Exception:
        return []


# Признаки того, что сервер упал именно из-за видеокарты, а не из-за
# настроек или сети. Для них есть обходной путь — расчёт на процессоре.
_GPU_FAILURE = ("ptx", "cuda", "cublas", "terminated", "0xc0000409",
                "out of memory", "kernel image")


def _is_gpu_failure(note: str) -> bool:
    low = (note or "").lower()
    return any(m in low for m in _GPU_FAILURE)


def _explain(llm_mod, cfg, note: str) -> str:
    """Человеческое объяснение вместо куска чужого лога."""
    try:
        text, _fatal = llm_mod.diagnose(RuntimeError(note), cfg)
        return text
    except Exception:
        return f"Сервер не ответил: {note[:200]}"


def make_cpu_model(model: str) -> Optional[str]:
    """Создаёт вариант модели, считающий на процессоре.

    Переменную окружения вроде CUDA_VISIBLE_DEVICES трогать нельзя: она
    глобальная и отобрала бы видеокарту у распознавания речи, которому
    она нужнее. Поэтому GPU отключается только для этой модели —
    параметром num_gpu в её описании.
    """
    target = f"{model.split(':')[0]}-cpu"
    body = f"FROM {model}\nPARAMETER num_gpu 0\n"
    path = os.path.join(tempfile.gettempdir(), "callqa_cpu.Modelfile")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
    except OSError as exc:
        say(f"    [!] Не удалось записать описание модели: {exc}")
        return None

    say(f"    создаю «{target}» (без видеокарты) ...")
    cmd, env = ollama_cmd("create", target, "-f", path)
    code, out = run(cmd or ["ollama", "create", target, "-f", path], timeout=900, env=env)
    try:
        os.remove(path)
    except OSError:
        pass
    if code != 0:
        say("    [!] Не получилось — см. extras_log.txt")
        say("        Остаётся обновить драйвер или отключить GPU в настройках")
        say("        самого сервера модели.")
        return None
    return target


def setup_local_llm(assume: Optional[bool] = None) -> bool:
    """Подключает локальную модель вместо платного API."""
    servers = find_local_llm()

    exe_cmd, _env = ollama_cmd()
    if not servers and exe_cmd:
        say(f"  Ollama найдена: {exe_cmd[0]} — запускаю ...")
        ok, note = start_ollama()
        if ok:
            servers = find_local_llm()
        else:
            say(f"    [!] {note}")

    if not servers:
        say("  Запущенных локальных LLM-серверов не найдено.")
        if not exe_cmd:
            say("  Ollama не установлена. Она даёт локальную модель без платы за запросы.")
            if IS_WIN and shutil.which("winget"):
                if ask("Установить Ollama через winget?", True, assume):
                    run(["winget", "install", "--id", "Ollama.Ollama", "-e",
                         "--accept-package-agreements", "--accept-source-agreements"],
                        timeout=1800)
                    say("    после установки перезапустите install_extras.bat")
                    return False
            else:
                say("  Скачать: https://ollama.com/download")
            return False
        say("  Ollama установлена, но не запущена. Запустите её и повторите:")
        say("      ollama serve")
        return False

    srv = servers[0]
    say(f"  Найден локальный сервер: {srv['name']} ({srv['base_url']})")
    models = srv.get("models") or []
    if models:
        say(f"  Доступные модели: {', '.join(models[:8])}")
    else:
        say("  Моделей пока не загружено.")
        say("  Рекомендации под вашу видеокарту:")
        for name, size, note in OLLAMA_MODELS:
            say(f"    {name:<16} {size:<9} {note}")
        if exe_cmd and ask(
                f"Загрузить {OLLAMA_MODELS[0][0]} сейчас?", True, assume):
            say(f"    выполняю: ollama pull {OLLAMA_MODELS[0][0]}")
            cmd, env = ollama_cmd("pull", OLLAMA_MODELS[0][0])
            code, _ = run(cmd, timeout=5400, env=env)
            if code != 0:
                say("    [!] Загрузка не удалась — см. extras_log.txt")
                return False
            models = [OLLAMA_MODELS[0][0]]
        else:
            return False

    model = models[0]
    for preferred, _s, _n in OLLAMA_MODELS:
        if preferred in models:
            model = preferred
            break

    if not ask(f"Использовать «{model}» для смысловых пунктов?", True, assume):
        return False

    try:
        sys.path.insert(0, str(ROOT))
        from callqa import llm as llm_mod
        from callqa.config import DEFAULT_CONFIG_PATH, Config
    except Exception as exc:
        say(f"  [!] Не читается конфиг: {exc}")
        return False

    current = os.path.join(os.path.dirname(DEFAULT_CONFIG_PATH), "current.yaml")
    path = current if os.path.exists(current) else DEFAULT_CONFIG_PATH
    cfg = Config.load(path)
    cfg.llm.enabled = True
    cfg.llm.provider = "openai"
    cfg.llm.base_url = srv["base_url"]
    cfg.llm.model = model
    cfg.llm.api_key = ""
    cfg.llm.compact = "auto"        # локальным моделям — укороченный запрос
    if srv.get("name") == "Ollama" and exe_cmd:
        cfg.llm.ollama_exe = exe_cmd[0]  # чтобы CallQA сам запускал именно её

    say("    проверяю ...")
    ok, note = llm_mod.check_connection(cfg)

    if not ok and _is_gpu_failure(note):
        say(f"    [!] {_explain(llm_mod, cfg, note)}")
        say()
        say("    Видеокарта GTX 10-й серии и старее часто не тянет свежие")
        say("    сборки: драйвер не принимает код, собранный новым CUDA.")
        say("    Лечится обновлением драйвера NVIDIA (для GTX 10xx — ветка 580,")
        say("    она последняя с поддержкой этих карт).")
        say()
        say("    Прямо сейчас можно обойтись без обновления: сделать вариант")
        say("    модели, считающий на процессоре. Медленнее, но работает.")
        cpu_model = None
        if exe_cmd and ask("Сделать CPU-вариант модели?", True, assume):
            cpu_model = make_cpu_model(model)
        if cpu_model:
            cfg.llm.model = cpu_model
            say(f"    проверяю «{cpu_model}» ...")
            ok, note = llm_mod.check_connection(cfg)

    if not ok:
        say(f"    [!] {_explain(llm_mod, cfg, note)}")
        say("    Чек-лист работает и без модели: смысловые пункты будут «н/д»")
        say("    и на балл не повлияют.")
        if not ask("Всё равно сохранить настройку?", False, None):
            return False
    else:
        say(f"    {note}")
    cfg.save(current)
    say(f"    записано в {current}")
    say("    Платный ключ теперь не нужен — смысловые пункты считает локальная модель.")
    return True


# --------------------------------------------------------------------------- #
#  torch
# --------------------------------------------------------------------------- #
def gpu_capability() -> Optional[float]:
    try:
        sys.path.insert(0, str(ROOT))
        from callqa.asr import compute_capability

        return compute_capability()
    except Exception:
        return None


def torch_plan(gpu: Dict[str, object]) -> Dict[str, object]:
    """Какой torch ставить под эту видеокарту.

    kind: cuda — сборка с CUDA; cpu — обычная; old_driver — карта подходит,
    но драйвер слишком старый. Для Pascal версия закреплена: свежий torch
    на GTX 10xx ставится, но видеокарту не видит.
    """
    if not gpu.get("present"):
        return {"kind": "cpu", "packages": ["torch", "torchaudio"], "extra": [],
                "why": "видеокарты NVIDIA нет"}
    driver = float(gpu.get("driver") or 0.0)
    cc = gpu_capability()
    if cc is not None and cc < 6.0:
        return {"kind": "cpu", "packages": ["torch", "torchaudio"], "extra": [],
                "cc": cc, "why": f"видеокарта слишком старая для torch с CUDA "
                                 f"(способность {cc})"}
    if cc is not None and cc < 7.0:
        plan = {"kind": "cuda", "tag": PASCAL_TORCH["tag"], "cc": cc,
                "packages": list(PASCAL_TORCH["packages"]),
                "extra": list(PASCAL_TORCH["extra"]), "pinned": True,
                "why": f"серия GTX 10 (способность {cc}): последний torch с её "
                       "поддержкой — 2.8.0+cu126"}
        if driver and driver < PASCAL_TORCH["min_driver"]:
            plan["kind"] = "old_driver"
            plan["why"] = (f"для torch 2.8.0+cu126 нужен драйвер "
                           f"{PASCAL_TORCH['min_driver']:.0f}+, сейчас {driver}")
        return plan
    tag = cuda_channel(driver) if driver else "cu126"
    if tag is None:
        return {"kind": "old_driver", "packages": ["torch", "torchaudio"], "extra": [],
                "cc": cc, "why": f"драйвер {driver} слишком старый для CUDA-сборок torch"}
    return {"kind": "cuda", "tag": tag, "cc": cc, "packages": ["torch", "torchaudio"],
            "extra": [], "pinned": False, "why": f"драйвер {driver} -> {tag}"}


def _pip_torch(plan: Dict[str, object]) -> int:
    args = list(plan["packages"])
    if plan.get("kind") == "cuda":
        args += ["--index-url", f"https://download.pytorch.org/whl/{plan['tag']}"]
    code = pip(*args)
    # torchcodec от свежего torch со старым не загрузится — ставим парную версию
    if code == 0 and plan.get("extra"):
        code = pip(*plan["extra"])
    return code


def install_torch(gpu: Dict[str, object], assume: Optional[bool] = None) -> bool:
    plan = torch_plan(gpu)
    if have("torch"):
        state = _cuda_works()
        if state == "CUDA_OK":
            say("  torch уже есть, видеокарта работает")
            return True
        if plan["kind"] == "cpu":
            say(f"  torch уже есть (режим CPU: {plan['why']})")
            return True
        if state == "NO_KERNEL":
            say("  torch видит видеокарту, но собран без её поддержки — "
                "эта версия torch для неё слишком новая.")
        else:
            say("  torch установлен, но не видит видеокарту.")

    say(f"  Видеокарта: {gpu.get('name') or '—'}; драйвер: {gpu.get('driver') or '—'}")
    say(f"  {plan['why']}")
    if plan["kind"] == "old_driver":
        say("  [!] Сначала обновите драйвер NVIDIA (nvidia.com/drivers), потом")
        say("      запустите:  install_extras.bat --torch")
        return False
    if plan["kind"] == "cuda":
        pkgs = " ".join(plan["packages"])
        say(f"  Поставлю {pkgs} ({plan['tag']}). Загрузка около 2.5 ГБ.")
        say("  Нужен только pyannote: разделение голосов в моно и база голосов")
        say("  операторов. Распознавание речи на видеокарте работает и без него.")
        if not ask("Ставить?", True, assume):
            return False
        code = _pip_torch(plan)
        state = _cuda_works() if code == 0 else ""
        if state == "CUDA_OK":
            say("    готово: torch работает на видеокарте")
            return True
        say("    [!] Сборка с CUDA не заработала — см. extras_log.txt. pyannote "
            "продолжит работать на процессоре.")
        return False

    if have("torch"):
        return True
    if not ask("Поставить torch для CPU (около 250 МБ)?", False, assume):
        return False
    if pip("torch", "torchaudio") != 0:
        say("    [!] Не установилось — см. extras_log.txt")
        return False
    say("    готово (режим CPU)")
    return True


def _cuda_works() -> str:
    """CUDA_OK — считает на видеокарте; NO_KERNEL — видит, но собран без её
    поддержки (свежий torch на GTX 10xx); CPU — не видит; "" — torch нет."""
    code, out = run([sys.executable, "-c",
                     "import torch\n"
                     "if not torch.cuda.is_available():\n print('CPU')\n"
                     "else:\n"
                     " try:\n"
                     "  (torch.ones(8, device='cuda') * 2).sum().item(); print('CUDA_OK')\n"
                     " except Exception as e:\n  print('NO_KERNEL', e)"],
                    timeout=300)
    for mark in ("CUDA_OK", "NO_KERNEL", "CPU"):
        if mark in out:
            return mark
    return ""


# --------------------------------------------------------------------------- #
#  pyannote + токен HuggingFace
# --------------------------------------------------------------------------- #
def install_pyannote(assume: Optional[bool] = None) -> bool:
    if have("pyannote.audio"):
        say("  pyannote уже установлен")
        return True
    say("  Диаризация (pyannote) — точное разделение голосов в МОНО-записях.")
    say("  Если ваши записи стерео (оператор и клиент в разных каналах),")
    say("  она не нужна: разделение и так точное.")
    if not ask("Ставить pyannote?", False, assume):
        return False
    if not have("torch"):
        say("    сначала нужен torch — ставлю его ...")
        plan = torch_plan(detect_gpu())
        if plan["kind"] != "cuda":
            plan = {"kind": "cpu", "packages": ["torch", "torchaudio"], "extra": []}
        if _pip_torch(plan) != 0:
            say("    [!] torch не установился, pyannote без него не работает")
            return False
    code = pip("pyannote.audio")
    if code != 0:
        say("    [!] Не установилось — см. extras_log.txt")
        return False
    say("    готово")
    return True


def setup_hf_token(assume: Optional[bool] = None) -> bool:
    """Записывает токен HuggingFace в конфиг — без него pyannote не скачает модель."""
    try:
        sys.path.insert(0, str(ROOT))
        from callqa.config import DEFAULT_CONFIG_PATH, Config
    except Exception as exc:
        say(f"  [!] Не читается конфиг: {exc}")
        return False

    current = os.path.join(os.path.dirname(DEFAULT_CONFIG_PATH), "current.yaml")
    path = current if os.path.exists(current) else DEFAULT_CONFIG_PATH
    cfg = Config.load(path)
    if cfg.speakers.hf_token:
        say("  Токен HuggingFace уже записан в настройках")
        return True

    say("  Токен HuggingFace нужен pyannote, чтобы скачать модель. Он бесплатный.")
    say("    1. huggingface.co -> зарегистрируйтесь")
    say("    2. примите условия моделей (кнопка Agree на странице модели):")
    say("       huggingface.co/pyannote/speaker-diarization-community-1")
    say("       huggingface.co/pyannote/speaker-diarization-3.1")
    say("       huggingface.co/pyannote/segmentation-3.0")
    say("       huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM  (база голосов)")
    say("    3. Settings -> Access Tokens -> New token (тип Read)")
    if assume is not None:
        say("  (пропускаю: токен нельзя получить автоматически)")
        return False
    if not ask("Ввести токен сейчас?", True, None):
        return False
    token = ask_secret("Токен (hf_...)")
    if not token:
        return False
    from callqa.config import clean_hf_token

    cfg.speakers.hf_token = clean_hf_token(token)
    cfg.save(current)
    say(f"    записан в {current}")
    return True


# --------------------------------------------------------------------------- #
#  Бесплатный облачный провайдер
# --------------------------------------------------------------------------- #
def setup_free_cloud(assume: Optional[bool] = None) -> bool:
    """Подключает провайдера с бесплатным тарифом (Groq, Gemini и т.п.)."""
    try:
        sys.path.insert(0, str(ROOT))
        from callqa import llm as llm_mod
        from callqa.config import DEFAULT_CONFIG_PATH, Config
    except Exception as exc:
        say(f"  [!] Не читается конфиг: {exc}")
        return False

    providers = llm_mod.FREE_PROVIDERS
    say("  Бесплатные тарифы, которые понимают тот же протокол:")
    say()
    for i, p in enumerate(providers, 1):
        say(f"    {i}. {p['name']:<16} {p['limits']}")
        say(f"       {p['note']}; регистрация: {p['signup']}")
    say()
    say("  ВАЖНО про приватность: на бесплатных тарифах стенограмма звонка")
    say("  уходит провайдеру, и многие оставляют право использовать её для")
    say("  обучения. Если в разговорах есть персональные данные клиентов —")
    say("  корректен только локальный вариант.")
    say()

    if assume is not None:
        say("  (пропускаю: ключ надо получить вручную)")
        return False
    if not ask("Настроить бесплатного провайдера?", False, None):
        return False

    try:
        choice = input(f"  Номер (1-{len(providers)}), Enter — отмена: ").strip()
    except EOFError:
        return False
    if not choice.isdigit() or not 1 <= int(choice) <= len(providers):
        return False
    p = providers[int(choice) - 1]

    say()
    say(f"  {p['name']}: получите ключ на {p['signup']}")
    say(f"  Лимиты бесплатного тарифа: {p['limits']}")
    key = ask_secret(f"Ключ для {p['name']}")
    if not key:
        return False

    current = os.path.join(os.path.dirname(DEFAULT_CONFIG_PATH), "current.yaml")
    path = current if os.path.exists(current) else DEFAULT_CONFIG_PATH
    cfg = Config.load(path)
    cfg.llm.enabled = True
    cfg.llm.provider = "openai"
    cfg.llm.base_url = p["base_url"]
    cfg.llm.model = p["model"]
    cfg.llm.api_key_env = p["key_env"]
    cfg.llm.api_key = key          # временно, только чтобы проверить

    say("    проверяю ключ ...")
    try:
        ok, note = llm_mod.check_connection(cfg)
    except Exception as exc:
        ok, note = False, str(exc)[:200]
    if not ok:
        say(f"    [!] Не подошло: {note}")
        say(f"        Проверьте, что модель «{p['model']}» доступна на вашем тарифе.")
        if not ask("Всё равно сохранить?", False, None):
            return False
    else:
        say(f"    {note}")

    cfg.llm.api_key = ""           # ключ в файле не храним
    cfg.save(current)
    if IS_WIN:
        code, _ = run(["setx", p["key_env"], key], timeout=60)
        if code == 0:
            say(f"    ключ сохранён в переменной {p['key_env']}")
            say("    (подхватится в новых окнах — это перезапустите)")
        else:
            cfg.llm.api_key = key
            cfg.save(current)
            say(f"    setx не сработал, ключ записан в {current}")
    else:
        cfg.llm.api_key = key
        cfg.save(current)
    say(f"    настройки записаны в {current}")
    return True


# --------------------------------------------------------------------------- #
#  Платный ключ
# --------------------------------------------------------------------------- #
def setup_llm_key(assume: Optional[bool] = None) -> bool:
    try:
        sys.path.insert(0, str(ROOT))
        from callqa.config import DEFAULT_CONFIG_PATH, Config
    except Exception as exc:
        say(f"  [!] Не читается конфиг: {exc}")
        return False

    current = os.path.join(os.path.dirname(DEFAULT_CONFIG_PATH), "current.yaml")
    path = current if os.path.exists(current) else DEFAULT_CONFIG_PATH
    cfg = Config.load(path)
    var = cfg.llm.api_key_env

    if cfg.llm_api_key():
        say(f"  Ключ уже задан (переменная {var})")
        return True

    say("  Ключ API нужен для смысловых пунктов чек-листа:")
    say("  «отработал возражение», «говорил о выгодах», «вежливый тон»,")
    say("  а также чтобы ловить перефразированные предложения товаров.")
    say("  Без него эти пункты помечаются «н/д» и НЕ портят балл —")
    say("  всё остальное считается полностью.")
    say()
    say("  Важно: подписка на Claude (Pro/Max) ключ API НЕ даёт —")
    say("  это разные продукты с раздельной оплатой.")
    say("    1. console.anthropic.com — регистрация (можно тем же аккаунтом)")
    say("    2. Billing — пополнить баланс")
    say("    3. API keys — Create key")
    say()
    say(f"  Оплата по факту, за токены. Текущая модель: {cfg.llm.model}")
    say("  Ориентир: около 1 цента за звонок на claude-sonnet-5,")
    say("  вдвое дешевле на claude-haiku-4-5 (меняется в настройках).")
    if assume is not None:
        say("  (пропускаю: ключ нельзя получить автоматически)")
        return False
    if not ask("Ввести ключ сейчас?", True, None):
        return False

    key = ask_secret(f"Ключ для {var} (sk-ant-...)")
    if not key:
        return False

    say("    проверяю ключ ...")
    cfg.llm.api_key = key
    try:
        from callqa import llm as llm_mod

        ok, note = llm_mod.check_connection(cfg)
    except Exception as exc:
        ok, note = False, str(exc)[:200]
    if not ok:
        say(f"    [!] Ключ не подошёл: {note}")
        if not ask("Всё равно сохранить?", False, None):
            return False
    else:
        say(f"    {note}")

    if IS_WIN:
        code, _ = run(["setx", var, key], timeout=60)
        if code == 0:
            say(f"    сохранён в переменной окружения {var}")
            say("    (подхватится в новых окнах — это окно перезапустите)")
        else:
            say("    [!] setx не сработал, записываю ключ в конфиг")
            cfg.save(current)
            say(f"    записан в {current}")
    else:
        cfg.save(current)
        say(f"    записан в {current}")
    return True


# --------------------------------------------------------------------------- #
def status() -> Dict[str, object]:
    gpu = detect_gpu()
    servers = find_local_llm()
    return {
        "ffmpeg": ffmpeg_present(),
        "gpu_hw": gpu,
        "cuda_asr": cuda_runtime_ready() if gpu["present"] else False,
        "local_llm": servers[0] if servers else None,
        "api_key": _api_key_present(),
        "torch": have("torch"),
        "pyannote": have("pyannote.audio"),
    }


def _api_key_present() -> bool:
    try:
        sys.path.insert(0, str(ROOT))
        from callqa.config import Config

        cfg = Config.load()
        return bool(cfg.llm.api_key or os.environ.get(cfg.llm.api_key_env, ""))
    except Exception:
        return False


def print_status(st: Dict[str, object]) -> None:
    gpu = st["gpu_hw"]
    srv = st["local_llm"]
    rows = [
        ("ffmpeg", bool(st["ffmpeg"]),
         str(st["ffmpeg"]) if st["ffmpeg"] else "нет — mp3/m4a/ogg не читаются"),
        ("видеокарта NVIDIA", bool(gpu["present"]),
         f"{gpu['name']}, драйвер {gpu['driver'] or '?'}" if gpu["present"]
         else "не найдена — распознавание пойдёт на процессоре"),
        ("CUDA для распознавания", bool(st["cuda_asr"]),
         "работает" if st["cuda_asr"] else
         ("библиотек нет — считается на процессоре" if gpu["present"]
          else "неприменимо")),
        ("локальная модель", bool(srv),
         f"{srv['name']}: {', '.join((srv.get('models') or ['моделей нет'])[:3])}"
         if srv else "не запущена — смысловые пункты будут «н/д»"),
        ("платный ключ API", bool(st["api_key"]),
         "задан" if st["api_key"] else "нет (не нужен, если есть локальная модель)"),
        ("pyannote", bool(st["pyannote"]),
         "есть" if st["pyannote"] else "нет — для моно упрощённое разделение голосов"),
    ]
    for name, ok, note in rows:
        say(f"    {name:<24} {'есть' if ok else 'нет ':<5} {note}")


def ensure_venv() -> None:
    """Перезапускает себя интерпретатором из .venv, если запустили не тем."""
    venv_py = ROOT / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")
    if not venv_py.is_file():
        return
    try:
        same = Path(sys.executable).resolve() == venv_py.resolve()
    except OSError:
        same = False
    if same:
        return
    say("  Перезапускаюсь окружением .venv ...")
    os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])


def main() -> int:
    ap = argparse.ArgumentParser(description="Доустановка компонентов CallQA")
    ap.add_argument("--all", action="store_true",
                    help="поставить всё, что можно, без вопросов")
    ap.add_argument("--check", action="store_true", help="только показать состояние")
    ap.add_argument("--torch", action="store_true",
                    help="только поставить torch под видеокарту (для pyannote)")
    ap.add_argument("--no-venv-switch", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.no_venv_switch:
        ensure_venv()

    if LOG.exists():
        LOG.unlink()
    log(f"CallQA extras; python: {sys.version}; exe: {sys.executable}")

    say("=" * 52)
    say("  CallQA — доустановка компонентов")
    say("=" * 52)
    say()
    say("  Сейчас:")
    st = status()
    print_status(st)
    say()

    if args.check:
        return 0

    if args.torch:
        say("-" * 52)
        say("  torch под видеокарту (для pyannote)")
        ok = install_torch(st["gpu_hw"], True if args.all else None)
        say()
        say("  Проверка:  run_cli.bat doctor")
        return 0 if ok else 1

    assume: Optional[bool] = True if args.all else None
    if args.all:
        say("  Режим --all: ставлю всё без вопросов (кроме ключей — их надо ввести).")
        say()

    say("-" * 52)
    say("  1. ffmpeg — чтение mp3, m4a, ogg")
    install_ffmpeg(assume)
    say()

    say("-" * 52)
    say("  2. Ускорение распознавания на видеокарте")
    if install_cuda_libs(st["gpu_hw"], assume):
        configure_gpu_settings(st["gpu_hw"])
    say()

    say("-" * 52)
    say("  3. Смысловые пункты чек-листа (необязательно)")
    say()
    say("  Штатный чек-лист работает БЕЗ всякой модели — он построен на")
    say("  правилах. Модель нужна, только если хотите, чтобы пункты понимали")
    say("  формулировки, которых нет в ваших списках фраз.")
    say()
    say("  Варианты по возрастанию издержек:")
    say("    A. локальная модель — бесплатно, ничего не уходит наружу")
    say("    Б. бесплатный тариф в облаке — бесплатно, но стенограмма уходит")
    say("    В. платный API — лучшее качество, около 1 цента за звонок")
    say()
    say("  --- A. Локальная модель ---")
    if not setup_local_llm(assume):
        say()
        say("  --- Б. Бесплатный тариф в облаке ---")
        if not setup_free_cloud(assume):
            say()
            say("  --- В. Платный ключ ---")
            setup_llm_key(assume)
    say()

    say("-" * 52)
    say("  4. Разделение голосов для моно-записей (pyannote)")
    if install_pyannote(assume):
        setup_hf_token(assume)
    say()

    if have("pyannote.audio") and st["gpu_hw"]["present"]:
        say("-" * 52)
        say("  5. pyannote на видеокарте (необязательно)")
        install_torch(st["gpu_hw"], assume)
        say()

    say("=" * 52)
    say("  Стало:")
    print_status(status())
    say("=" * 52)
    say()
    say("  Полная проверка:  run_cli.bat doctor")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\n  Прервано пользователем.")
        sys.exit(130)
