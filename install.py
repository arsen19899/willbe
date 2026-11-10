"""Установщик CallQA.

Запускается любым Python 3.8+ (в том числе 3.14): сам находит на машине
подходящий интерпретатор 3.10–3.12, собирает на нём виртуальное окружение
и ставит зависимости группами, проверяя каждый шаг по фактическому
результату, а не по коду возврата.

  python install.py            обычная установка
  python install.py --check    только показать найденные интерпретаторы
  python install.py --yes      не задавать вопросов
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
LOG = ROOT / "install_log.txt"

MIN_VER = (3, 10)
MAX_VER = (3, 12)
PREFERRED = "3.12"

# Группы зависимостей: (заголовок, пакеты, модули для проверки, обязательна ли)
GROUPS = [
    ("Основа: чтение файлов, отчёты, веб-интерфейс",
     ["numpy", "PyYAML", "openpyxl", "requests", "soundfile",
      "fastapi", "uvicorn", "python-multipart"],
     ["numpy", "yaml", "openpyxl", "requests", "soundfile",
      "fastapi", "uvicorn", "multipart"],
     True),
    ("Распознавание речи (faster-whisper, ~200 МБ)",
     ["faster-whisper"],
     ["faster_whisper"],
     True),
    ("Морфология русского и нечёткий поиск",
     ["rapidfuzz", "pymorphy3", "pymorphy3-dicts-ru", "scipy"],
     ["rapidfuzz", "pymorphy3"],
     False),
]

IS_WIN = os.name == "nt"


# --------------------------------------------------------------------------- #
#  Вывод
# --------------------------------------------------------------------------- #
def say(msg: str = "") -> None:
    print(msg, flush=True)


def log(text: str) -> None:
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        f.write(text.rstrip() + "\n")


def run(cmd: List[str], timeout: int = 3600) -> Tuple[int, str]:
    """Запускает команду, пишет весь вывод в лог, возвращает (код, вывод)."""
    log(f"\n$ {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError:
        log("  -> команда не найдена")
        return 127, ""
    except subprocess.TimeoutExpired:
        log("  -> превышено время ожидания")
        return 124, ""
    out = (p.stdout or "") + (p.stderr or "")
    log(out)
    return p.returncode, out


# --------------------------------------------------------------------------- #
#  Поиск подходящего интерпретатора
# --------------------------------------------------------------------------- #
PROBE = ("import sys;print('PYOK%d.%d' % (sys.version_info[0], sys.version_info[1]))")


def probe(cmd: List[str]) -> Optional[Tuple[int, int]]:
    """Возвращает версию интерпретатора, только если он действительно её напечатал.

    Коду возврата не доверяем: новый Python Install Manager отдаёт 0
    даже когда нужного рантайма нет.
    """
    try:
        p = subprocess.run(cmd + ["-c", PROBE], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=30)
    except Exception:
        return None
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("PYOK"):
            try:
                major, minor = line[4:].split(".")[:2]
                return int(major), int(minor)
            except ValueError:
                return None
    return None


def candidates() -> List[List[str]]:
    out: List[List[str]] = []

    def add(c: List[str]) -> None:
        if c not in out:
            out.append(c)

    if IS_WIN:
        for v in ("3.12", "3.11", "3.10"):
            add(["py", f"-{v}"])
    else:
        for v in ("3.12", "3.11", "3.10"):
            exe = shutil.which(f"python{v}")
            if exe:
                add([exe])

    # текущий интерпретатор — вдруг он и так подходит
    add([sys.executable])
    for name in ("python", "python3"):
        exe = shutil.which(name)
        if exe:
            add([exe])

    if IS_WIN:
        local = Path(os.environ.get("LOCALAPPDATA", "") or "C:/")
        roots = [
            local / "Programs" / "Python",
            local / "Microsoft" / "WindowsApps" / "PythonSoftwareFoundation",
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")),
            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")),
            Path("C:/"),
        ]
        for root in roots:
            try:
                if not root.is_dir():
                    continue
                for child in sorted(root.iterdir(), reverse=True):
                    if not child.is_dir():
                        continue
                    if not child.name.lower().startswith("python3"):
                        continue
                    exe = child / "python.exe"
                    if exe.is_file():
                        add([str(exe)])
            except OSError:
                continue
        for c in _from_registry():
            add(c)

    return out


def _from_registry() -> List[List[str]]:
    """Пути, зарегистрированные установщиками Python в реестре Windows."""
    found: List[List[str]] = []
    try:
        import winreg
    except ImportError:
        return found
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (0, getattr(winreg, "KEY_WOW64_32KEY", 0)):
            try:
                key = winreg.OpenKey(hive, r"SOFTWARE\Python\PythonCore", 0,
                                     winreg.KEY_READ | view)
            except OSError:
                continue
            try:
                i = 0
                while True:
                    try:
                        ver = winreg.EnumKey(key, i)
                    except OSError:
                        break
                    i += 1
                    try:
                        with winreg.OpenKey(key, ver + r"\InstallPath", 0,
                                            winreg.KEY_READ | view) as k2:
                            path, _ = winreg.QueryValueEx(k2, "ExecutablePath")
                            if path and Path(path).is_file():
                                found.append([str(path)])
                    except OSError:
                        continue
            finally:
                key.Close()
    return found


def find_python(verbose: bool = True) -> Tuple[Optional[List[str]], List[str]]:
    """Возвращает (команда подходящего интерпретатора, список отвергнутых версий)."""
    rejected: List[str] = []
    seen_versions = set()
    for cmd in candidates():
        ver = probe(cmd)
        if ver is None:
            continue
        label = f"{ver[0]}.{ver[1]}"
        if MIN_VER <= ver <= MAX_VER:
            if verbose:
                say(f"  Найден Python {label}: {' '.join(cmd)}")
            return cmd, rejected
        if label not in seen_versions:
            seen_versions.add(label)
            rejected.append(f"{label} ({' '.join(cmd)})")
    return None, rejected


# --------------------------------------------------------------------------- #
def explain_version_problem(rejected: List[str]) -> None:
    say()
    say("  Не найден подходящий Python.")
    say()
    if rejected:
        say("  Установлены, но не подходят:")
        for r in rejected:
            say(f"    - {r}")
        say()
    say(f"  Нужен Python {MIN_VER[0]}.{MIN_VER[1]}–{MAX_VER[0]}.{MAX_VER[1]}.")
    say()
    say("  Python 3.13 и 3.14 слишком новые: у ctranslate2 (движок")
    say("  распознавания речи) и pymorphy3 пока нет готовых сборок под них,")
    say("  pip пытается собрать их из исходников и падает.")
    say()


def offer_py_install(assume_yes: bool) -> bool:
    """Предлагает установить рантайм через новый Python Install Manager."""
    if not IS_WIN or not shutil.which("py"):
        return False
    say("  Ваш лаунчер «py» умеет сам скачивать нужные версии.")
    if not assume_yes:
        try:
            answer = input(f"  Установить Python {PREFERRED} сейчас? [y/N]: ").strip().lower()
        except EOFError:
            return False
        if answer not in ("y", "yes", "д", "да"):
            return False
    say()
    say(f"  Выполняю: py install {PREFERRED}")
    say("  (загрузка около 30 МБ, займёт минуту)")
    say()
    code, _ = run(["py", "install", PREFERRED], timeout=1800)
    say()
    return True


def manual_instructions() -> None:
    say("  Установите Python 3.12 и запустите установку заново.")
    say()
    say("    Вариант А — через лаунчер, в любом терминале:")
    say(f"        py install {PREFERRED}")
    say()
    say("    Вариант Б — обычный установщик:")
    say("        https://www.python.org/downloads/release/python-31210/")
    say('        файл «Windows installer (64-bit)»')
    say('        отметьте «Add python.exe to PATH»')
    say()
    say("  Удалять Python 3.14 не нужно — версии живут рядом.")
    say()


# --------------------------------------------------------------------------- #
def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def make_venv(py: List[str]) -> bool:
    if VENV.exists():
        say("  Удаляю прежнее окружение .venv ...")
        shutil.rmtree(VENV, ignore_errors=True)
    say("  Создаю виртуальное окружение ...")
    run(py + ["-m", "venv", str(VENV)])
    # проверяем по факту, а не по коду возврата
    if not venv_python().is_file():
        say()
        say("  [ОШИБКА] Окружение не создалось: нет файла")
        say(f"           {venv_python()}")
        say()
        say("  Обычная причина: лаунчер «py» объявил версию, которая на самом")
        say("  деле не установлена. Поставьте рантайм явно:")
        say(f"        py install {PREFERRED}")
        say()
        say(f"  Точная ошибка — в начале файла {LOG.name}")
        return False
    ver = probe([str(venv_python())])
    say(f"  Окружение готово (Python {ver[0]}.{ver[1]})" if ver else "  Окружение готово")
    return True


def pip_install(vpy: Path, packages: List[str], modules: List[str]) -> bool:
    run([str(vpy), "-m", "pip", "install", *packages])
    code, _ = run([str(vpy), "-c", "import " + ",".join(modules)], timeout=180)
    return code == 0


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    say()
    say("  [!] ffmpeg не найден в PATH.")
    say("      Без него не читаются mp3, m4a, ogg. WAV работает.")
    say("      Скачать: https://www.gyan.dev/ffmpeg/builds/ (release essentials)")
    say("      Распакуйте и добавьте папку с ffmpeg.exe в переменную Path.")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Установка CallQA")
    ap.add_argument("--check", action="store_true",
                    help="только показать найденные интерпретаторы")
    ap.add_argument("--yes", "-y", action="store_true", help="не задавать вопросов")
    args = ap.parse_args()

    if LOG.exists():
        LOG.unlink()
    log(f"CallQA installer; загрузчик: {sys.version}")
    log(f"платформа: {sys.platform}; папка: {ROOT}")

    say("=" * 46)
    say("  CallQA — установка")
    say("=" * 46)
    say()

    if args.check:
        say("  Проверяю интерпретаторы на этой машине:")
        any_found = False
        for cmd in candidates():
            ver = probe(cmd)
            if ver:
                any_found = True
                mark = "подходит" if MIN_VER <= ver <= MAX_VER else "слишком новый"
                if ver < MIN_VER:
                    mark = "слишком старый"
                say(f"    {ver[0]}.{ver[1]:<4} {mark:<16} {' '.join(cmd)}")
            else:
                say(f"    {'—':<5} {'не отвечает':<16} {' '.join(cmd)}")
        if not any_found:
            say("    (рабочих интерпретаторов не найдено)")
        say()
        say(f"  ffmpeg: {shutil.which('ffmpeg') or 'не найден'}")
        return 0

    py, rejected = find_python()
    if py is None:
        explain_version_problem(rejected)
        if offer_py_install(args.yes):
            py, rejected = find_python()
        if py is None:
            manual_instructions()
            return 1

    say()
    if not make_venv(py):
        return 1

    vpy = venv_python()
    say()
    run([str(vpy), "-m", "pip", "install", "--upgrade", "pip"])

    failed_required = False
    for i, (title, packages, modules, required) in enumerate(GROUPS, 1):
        say(f"  [{i}/{len(GROUPS)}] {title} ...")
        if pip_install(vpy, packages, modules):
            say("          готово")
        elif required:
            say(f"          [ОШИБКА] не установилось — подробности в {LOG.name}")
            failed_required = True
        else:
            say("          [!] не установилось. Программа будет работать,")
            say("              но станет искать фразы только дословно.")

    check_ffmpeg()
    say()

    if failed_required:
        say("=" * 46)
        say("  Установка завершилась с ошибками.")
        say(f"  Пришлите файл {LOG.name} — разберёмся.")
        say("=" * 46)
        return 1

    say("=" * 46)
    say("  Готово.")
    say()
    say("    run_web.bat           — веб-интерфейс")
    say("    run_cli.bat doctor    — проверка окружения")
    say("    run_cli.bat demo      — пример отчёта без записей")
    say("=" * 46)
    say()
    subprocess.run([str(vpy), "-m", "callqa.cli", "doctor"], cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\n  Прервано пользователем.")
        sys.exit(130)
