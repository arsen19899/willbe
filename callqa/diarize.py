"""Разделение по голосам для моно-записей.

Основной путь — pyannote.audio (нужен токен HuggingFace). Если он недоступен,
работает встроенный запасной вариант: MFCC-подобные признаки + кластеризация
на 2 группы. Он заметно грубее, поэтому в отчёт добавляется предупреждение.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .models import Segment

Turn = Tuple[float, float, str]  # start, end, speaker_label


def diarize(samples: np.ndarray, sr: int, cfg) -> Tuple[List[Turn], str, Optional[str]]:
    """Возвращает (интервалы, метод, предупреждение)."""
    turns, warn = _pyannote(samples, sr, cfg)
    if turns:
        return turns, "pyannote", warn
    turns = _fallback_cluster(samples, sr, cfg)
    return (
        turns,
        "fallback",
        "Диаризация выполнена упрощённым алгоритмом (pyannote недоступен). "
        "Разделение оператор/клиент и подсчёт перебиваний менее точны. "
        "Для точного результата используйте стереозаписи или укажите speakers.hf_token.",
    )


def _pyannote(samples: np.ndarray, sr: int, cfg):
    token = (cfg.hf_token or "").strip()
    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception:
        return [], None
    try:
        kwargs = {"use_auth_token": token} if token else {}
        pipeline = Pipeline.from_pretrained(cfg.diarization_model, **kwargs)
        if pipeline is None:
            return [], None
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        wav = torch.from_numpy(np.ascontiguousarray(samples)).float().unsqueeze(0)
        ann = pipeline({"waveform": wav, "sample_rate": sr},
                       num_speakers=cfg.num_speakers or None)
        turns = [(float(seg.start), float(seg.end), str(label))
                 for seg, _, label in ann.itertracks(yield_label=True)]
        turns.sort(key=lambda t: t[0])
        return turns, None
    except Exception as exc:
        return [], f"pyannote не отработал: {exc}"


# --------------------------------------------------------------------------- #
#  Запасной вариант без внешних моделей
# --------------------------------------------------------------------------- #
def _fallback_cluster(samples: np.ndarray, sr: int, cfg, win_sec: float = 1.0):
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    n = int(sr * win_sec)
    if len(samples) < n * 4:
        return [(0.0, len(samples) / sr, "SPK_0")]

    frames = len(samples) // n
    x = samples[: frames * n].reshape(frames, n)
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=1) + 1e-12)
    active = rms > max(1e-4, np.percentile(rms, 35))

    feats = np.array([_spectral_feats(x[i], sr) for i in range(frames)])
    if active.sum() < 4:
        return [(0.0, frames * win_sec, "SPK_0")]

    f_act = feats[active]
    mu, sd = f_act.mean(axis=0), f_act.std(axis=0) + 1e-9
    z = (feats - mu) / sd

    labels = _kmeans2(z[active], seed=0)
    full = np.full(frames, -1, dtype=int)
    full[active] = labels

    turns: List[Turn] = []
    cur, start = None, 0.0
    for i in range(frames):
        lab = full[i]
        name = f"SPK_{lab}" if lab >= 0 else None
        if name != cur:
            if cur is not None:
                turns.append((start, i * win_sec, cur))
            cur, start = name, i * win_sec
    if cur is not None:
        turns.append((start, frames * win_sec, cur))
    return [t for t in turns if t[1] - t[0] >= 0.4]


def _spectral_feats(frame: np.ndarray, sr: int) -> np.ndarray:
    w = frame * np.hanning(len(frame))
    spec = np.abs(np.fft.rfft(w)) + 1e-12
    freqs = np.fft.rfftfreq(len(frame), 1 / sr)
    power = spec ** 2
    total = power.sum()
    centroid = float((freqs * power).sum() / total)
    spread = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / total))
    rolloff = float(freqs[np.searchsorted(np.cumsum(power), 0.85 * total)]
                    if total > 0 else 0.0)
    flat = float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-12))
    bands = []
    edges = [0, 300, 600, 1000, 1800, 3000, 5000, 8000]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (freqs >= a) & (freqs < b)
        bands.append(float(np.log(power[m].sum() + 1e-12)) if m.any() else 0.0)
    return np.array([centroid, spread, rolloff, flat] + bands)


def _kmeans2(z: np.ndarray, seed: int = 0, iters: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # инициализация двумя максимально далёкими точками
    i0 = int(rng.integers(len(z)))
    d = np.linalg.norm(z - z[i0], axis=1)
    i1 = int(np.argmax(d))
    c = np.stack([z[i0], z[i1]])
    labels = np.zeros(len(z), dtype=int)
    for _ in range(iters):
        d = np.stack([np.linalg.norm(z - c[k], axis=1) for k in range(2)], axis=1)
        new = np.argmin(d, axis=1)
        if (new == labels).all():
            break
        labels = new
        for k in range(2):
            if (labels == k).any():
                c[k] = z[labels == k].mean(axis=0)
    return labels


# --------------------------------------------------------------------------- #
def assign_turns_to_segments(segments: List[Segment], turns: List[Turn]) -> List[Segment]:
    """Проставляет сегментам метку спикера по максимальному перекрытию с интервалами."""
    if not turns:
        for s in segments:
            s.speaker = "SPK_0"
        return segments

    for seg in segments:
        best, best_ov = None, 0.0
        for st, en, lab in turns:
            ov = min(seg.end, en) - max(seg.start, st)
            if ov > best_ov:
                best, best_ov = lab, ov
        seg.speaker = best or "SPK_0"

    # то же самое для отдельных слов: сегмент может содержать реплики обоих
    out: List[Segment] = []
    for seg in segments:
        if not seg.words:
            out.append(seg)
            continue
        pieces: List[Segment] = []
        cur_lab, buf = None, []
        for w in seg.words:
            lab, ov = seg.speaker, 0.0
            for st, en, l in turns:
                o = min(w.end, en) - max(w.start, st)
                if o > ov:
                    lab, ov = l, o
            if lab != cur_lab and buf:
                pieces.append(_mk(cur_lab, buf))
                buf = []
            cur_lab = lab
            buf.append(w)
        if buf:
            pieces.append(_mk(cur_lab, buf))
        out.extend(pieces if pieces else [seg])
    return sorted(out, key=lambda s: s.start)


def _mk(label: str, words) -> Segment:
    return Segment(label or "SPK_0", words[0].start, words[-1].end,
                   " ".join(w.text for w in words).strip(), list(words))
