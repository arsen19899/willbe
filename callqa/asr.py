"""Распознавание речи через faster-whisper.

Стереозапись распознаётся по каналам отдельно — это даёт идеальное разделение
оператор/клиент и честный подсчёт перебиваний. Моно распознаётся целиком,
а роли расставляются диаризацией (см. diarize.py).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from typing import Callable, List, Optional

import numpy as np

from .audio import LoadedAudio, channel, to_mono
from .config import ASRConfig
from .models import Segment, Transcript, Word

_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()

ProgressCb = Optional[Callable[[str, float], None]]


class ASRUnavailable(RuntimeError):
    pass


def _pick_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch  # noqa

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def compute_capability() -> Optional[float]:
    """Вычислительная способность видеокарты — без установленного torch.

    От неё зависит, какие типы вычислений вообще доступны:
    float16 требует 7.0 и выше, на Pascal (6.1, серия GTX 10xx) его нет.
    """
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return float(f"{major}.{minor}")
    except Exception:
        pass

    exe = shutil.which("nvidia-smi")
    if not exe and os.name == "nt":
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                            "System32", "nvidia-smi.exe")
        if os.path.isfile(cand):
            exe = cand
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().splitlines()[0].strip())
    except Exception:
        pass
    return None


def _pick_compute_type(requested: str, device: str) -> str:
    if requested and requested != "auto":
        return requested
    if device != "cuda":
        return "int8"
    cc = compute_capability()
    if cc is None:
        return "int8_float32"      # безопасно на любой видеокарте
    if cc >= 7.0:
        return "float16"           # Turing и новее — тензорные ядра
    if cc >= 6.1:
        return "int8_float32"      # Pascal: float16 не поддерживается
    return "float32"


def _load_hint(model_name: str, exc: Exception) -> str:
    msg = str(exc)
    tip = ""
    low = msg.lower()
    if any(w in low for w in ("403", "proxy", "connection", "timeout", "resolve",
                              "network", "ssl", "certificate")):
        tip = ("\nПохоже на проблему с доступом в интернет: при первом запуске модель "
               "скачивается с huggingface.co. Скачайте её один раз на машине с сетью "
               "и укажите путь к папке модели в asr.model, либо задайте "
               "asr.download_root и переменную HF_HOME.")
    elif "out of memory" in low or "cuda" in low:
        tip = ("\nНе хватило видеопамяти. Попробуйте модель поменьше (medium/small) "
               "или asr.device: cpu.")
    return f"Не удалось загрузить модель '{model_name}': {msg}{tip}"


def load_model(cfg: ASRConfig):
    """Ленивая загрузка и кэширование модели whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise ASRUnavailable(
            "Не установлен faster-whisper. Выполните: pip install faster-whisper"
        ) from exc

    device = _pick_device(cfg.device)
    compute = _pick_compute_type(cfg.compute_type, device)
    key = (cfg.model, device, compute, cfg.download_root, cfg.cpu_threads)

    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        kwargs = dict(device=device, compute_type=compute)
        if cfg.download_root:
            kwargs["download_root"] = cfg.download_root
        if cfg.cpu_threads:
            kwargs["cpu_threads"] = cfg.cpu_threads
        rest = {k: v for k, v in kwargs.items() if k not in ("device", "compute_type")}
        try:
            model = WhisperModel(cfg.model, **kwargs)
        except Exception as exc:
            if device == "cuda":
                # на части сборок нет cudnn/CUDA-рантайма — тихо откатываемся на CPU
                try:
                    model = WhisperModel(cfg.model, device="cpu",
                                         compute_type="int8", **rest)
                except Exception as exc2:
                    raise ASRUnavailable(_load_hint(cfg.model, exc2)) from exc2
            else:
                raise ASRUnavailable(_load_hint(cfg.model, exc)) from exc
        _MODEL_CACHE[key] = model
        return model


def _transcribe_array(model, samples: np.ndarray, cfg: ASRConfig,
                      speaker: str) -> List[Segment]:
    kwargs = dict(
        language=cfg.language or None,
        beam_size=cfg.beam_size,
        vad_filter=cfg.vad_filter,
        word_timestamps=cfg.word_timestamps,
        condition_on_previous_text=False,
    )
    if cfg.initial_prompt:
        kwargs["initial_prompt"] = cfg.initial_prompt
    if cfg.vad_filter:
        kwargs["vad_parameters"] = dict(min_silence_duration_ms=400)

    segments, _info = model.transcribe(samples.astype(np.float32), **kwargs)

    out: List[Segment] = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        words: List[Word] = []
        for w in (getattr(s, "words", None) or []):
            wt = (w.word or "").strip()
            if wt:
                words.append(Word(wt, float(w.start), float(w.end)))
        out.append(Segment(speaker, float(s.start), float(s.end), text, words))
    return out


def transcribe(audio: LoadedAudio, cfg: ASRConfig, speakers_cfg,
               progress: ProgressCb = None) -> Transcript:
    """Возвращает транскрипт с уже проставленными ролями (для стерео) или
    с общим спикером 'unknown' (для моно — роли назначит diarize/roles)."""
    model = load_model(cfg)
    name = os.path.basename(audio.path)

    use_stereo = (
        speakers_cfg.mode in ("auto", "stereo")
        and audio.channels >= 2
        and not audio.dual_mono
    )

    if use_stereo:
        segs: List[Segment] = []
        for idx, role in ((0, speakers_cfg.left), (1, speakers_cfg.right)):
            if progress:
                progress(f"{name}: распознаю дорожку {idx + 1}/2", 0.1 + idx * 0.4)
            segs.extend(_transcribe_array(model, channel(audio, idx), cfg, role))
        segs.sort(key=lambda s: s.start)
        source = "stereo"
    else:
        if progress:
            progress(f"{name}: распознаю речь", 0.15)
        segs = _transcribe_array(model, to_mono(audio), cfg, "unknown")
        source = "single"

    return Transcript(
        segments=segs,
        duration=audio.duration,
        language=cfg.language,
        channels=audio.channels,
        source=source,
        asr_model=cfg.model,
    )
