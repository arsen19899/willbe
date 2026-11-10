"""Загрузка аудио, определение стерео/моно, подготовка каналов для ASR."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".aac", ".wma",
             ".amr", ".gsm", ".webm", ".mp4", ".mkv", ".avi", ".aiff", ".au"}

TARGET_SR = 16000


@dataclass
class LoadedAudio:
    samples: np.ndarray        # (n,) моно или (n, 2) стерео, float32 [-1, 1]
    sr: int
    channels: int
    duration: float
    path: str
    dual_mono: bool = False    # каналы почти идентичны -> фактически моно


def _local_bin() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin"
    )


def ffmpeg_path() -> Optional[str]:
    """Ищет ffmpeg: сначала в bin/ проекта, затем в PATH и FFMPEG_BINARY."""
    for name in ("ffmpeg.exe", "ffmpeg"):
        p = os.path.join(_local_bin(), name)
        if os.path.isfile(p):
            return p
    return shutil.which("ffmpeg") or os.environ.get("FFMPEG_BINARY")


def is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXT


def list_audio_files(root: str, recursive: bool = True) -> List[str]:
    if os.path.isfile(root):
        return [root] if is_audio_file(root) else []
    out: List[str] = []
    if recursive:
        for dirpath, _dirs, files in os.walk(root):
            for f in sorted(files):
                p = os.path.join(dirpath, f)
                if is_audio_file(p):
                    out.append(p)
    else:
        for f in sorted(os.listdir(root)):
            p = os.path.join(root, f)
            if os.path.isfile(p) and is_audio_file(p):
                out.append(p)
    return out


# --------------------------------------------------------------------------- #
def load_audio(path: str, dual_mono_corr: float = 0.98,
               target_sr: int = TARGET_SR) -> LoadedAudio:
    """Читает файл в float32. Сначала soundfile, при неудаче — ffmpeg."""
    data, sr = _read_soundfile(path)
    if data is None:
        data, sr = _read_ffmpeg(path, target_sr)
    if data is None:
        raise RuntimeError(
            f"Не удалось прочитать аудио: {os.path.basename(path)}. "
            "Установите ffmpeg и добавьте его в PATH."
        )

    if data.ndim == 1:
        data = data[:, None]
    channels = data.shape[1]

    if sr != target_sr:
        data = _resample(data, sr, target_sr)
        sr = target_sr

    dual_mono = False
    if channels >= 2:
        left, right = data[:, 0], data[:, 1]
        if _rms(left) < 1e-5 or _rms(right) < 1e-5:
            dual_mono = True  # одна дорожка молчит — стерео бесполезно
        else:
            corr = _corr(left, right)
            dual_mono = corr >= dual_mono_corr

    if channels > 2:
        data = data[:, :2]
        channels = 2

    out = data[:, 0] if channels == 1 else data
    return LoadedAudio(
        samples=np.ascontiguousarray(out.astype(np.float32)),
        sr=sr,
        channels=channels,
        duration=len(data) / float(sr),
        path=path,
        dual_mono=dual_mono,
    )


def to_mono(a: LoadedAudio) -> np.ndarray:
    if a.samples.ndim == 1:
        return a.samples
    return a.samples.mean(axis=1).astype(np.float32)


def channel(a: LoadedAudio, idx: int) -> np.ndarray:
    if a.samples.ndim == 1:
        return a.samples
    return np.ascontiguousarray(a.samples[:, idx].astype(np.float32))


def write_wav(samples: np.ndarray, sr: int, path: str) -> str:
    x = np.clip(samples, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def temp_wav(samples: np.ndarray, sr: int, suffix: str = ".wav") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix, prefix="callqa_")
    os.close(fd)
    return write_wav(samples, sr, p)


def silence_ratio(samples: np.ndarray, sr: int, frame_ms: int = 30,
                  threshold_db: float = -45.0) -> float:
    """Доля кадров ниже порога громкости. Грубая, но достаточная оценка «пустоты»."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    n = max(1, int(sr * frame_ms / 1000))
    frames = len(samples) // n
    if frames == 0:
        return 1.0
    x = samples[: frames * n].reshape(frames, n)
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    return float(np.mean(db < threshold_db))


# --------------------------------------------------------------------------- #
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b), 16000 * 120)  # хватит двух минут
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den < 1e-12:
        return 1.0
    return float(abs((a * b).sum() / den))


def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data
    try:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr_in, sr_out)
        return resample_poly(data, sr_out // g, sr_in // g, axis=0).astype(np.float32)
    except Exception:
        n_out = int(round(len(data) * sr_out / sr_in))
        idx = np.linspace(0, len(data) - 1, n_out)
        out = np.stack([np.interp(idx, np.arange(len(data)), data[:, c])
                        for c in range(data.shape[1])], axis=1)
        return out.astype(np.float32)


def _read_soundfile(path: str) -> Tuple[Optional[np.ndarray], int]:
    try:
        import soundfile as sf

        data, sr = sf.read(path, always_2d=True, dtype="float32")
        return data, sr
    except Exception:
        return None, 0


def _read_ffmpeg(path: str, target_sr: int) -> Tuple[Optional[np.ndarray], int]:
    exe = ffmpeg_path()
    if not exe:
        return None, 0
    # сначала пытаемся сохранить исходное число каналов
    for ch in (2, 1):
        cmd = [exe, "-v", "error", "-i", path, "-f", "f32le",
               "-acodec", "pcm_f32le", "-ar", str(target_sr), "-ac", str(ch), "-"]
        try:
            proc = subprocess.run(cmd, capture_output=True, check=False)
        except Exception:
            return None, 0
        if proc.returncode == 0 and proc.stdout:
            arr = np.frombuffer(proc.stdout, dtype="<f4")
            if ch == 2:
                if len(arr) % 2:
                    arr = arr[:-1]
                arr = arr.reshape(-1, 2)
            else:
                arr = arr.reshape(-1, 1)
            return arr.astype(np.float32), target_sr
    return None, 0


def ffprobe_path() -> Optional[str]:
    for name in ("ffprobe.exe", "ffprobe"):
        p = os.path.join(_local_bin(), name)
        if os.path.isfile(p):
            return p
    return shutil.which("ffprobe")


def probe_channels(path: str) -> int:
    """Быстро узнать число каналов без полной загрузки (если есть ffprobe)."""
    exe = ffprobe_path()
    if not exe:
        return 0
    cmd = [exe, "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=channels", "-of", "csv=p=0", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip() or 0)
    except Exception:
        return 0
