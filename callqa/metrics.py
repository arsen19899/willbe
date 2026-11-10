"""Метрики диалога: перебивания, доля речи, паузы, монологи, темп речи."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .config import Config, MetricsConfig
from .models import CLIENT, OPERATOR, Evidence, Segment, Transcript
from .text import SpeechIndex, find_pattern, normalize, tokenize

# Человекочитаемые названия метрик — используются в отчёте и в UI
METRIC_TITLES = {
    "duration": "Длительность, сек",
    "operator_talk_sec": "Речь оператора, сек",
    "client_talk_sec": "Речь клиента, сек",
    "operator_talk_ratio": "Доля речи оператора",
    "client_talk_ratio": "Доля речи клиента",
    "silence_ratio": "Доля тишины",
    "operator_interruptions": "Перебиваний оператором",
    "client_interruptions": "Перебиваний клиентом",
    "total_interruptions": "Перебиваний всего",
    "overlap_sec": "Наложение речи, сек",
    "long_pauses": "Затяжных пауз",
    "longest_pause_sec": "Самая длинная пауза, сек",
    "operator_monologues": "Монологов оператора",
    "longest_monologue_sec": "Самый длинный монолог, сек",
    "operator_wpm": "Темп оператора, слов/мин",
    "client_wpm": "Темп клиента, слов/мин",
    "operator_words": "Слов у оператора",
    "client_words": "Слов у клиента",
    "turns": "Реплик всего",
    "stop_phrase_hits": "Стоп-фраз найдено",
    "profanity_hits": "Нежелательной лексики",
    "client_questions": "Вопросов от клиента",
    "operator_questions": "Вопросов от оператора",
    "response_delay_avg": "Средняя задержка ответа оператора, сек",
}


def _merge_spans(segs: List[Segment]) -> List[Tuple[float, float]]:
    spans = sorted(((s.start, s.end) for s in segs if s.end > s.start))
    out: List[Tuple[float, float]] = []
    for st, en in spans:
        if out and st <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], en))
        else:
            out.append((st, en))
    return out


def _total(spans: List[Tuple[float, float]]) -> float:
    return sum(en - st for st, en in spans)


def _is_backchannel(seg: Segment, mcfg: MetricsConfig) -> bool:
    if not mcfg.interruption_ignore_backchannel:
        return False
    toks = tokenize(seg.text)
    if not toks:
        return True
    if len(toks) > 3:
        return False
    bc = {normalize(w) for w in (mcfg.backchannel_words or [])}
    return all(t in bc for t in toks)


def detect_interruptions(transcript: Transcript, mcfg: MetricsConfig
                         ) -> Tuple[List[Dict[str, Any]], float]:
    """Перебивание = сторона начинает говорить, пока другая ещё говорит,
    наложение достаточно длинное, реплика содержательная, и говорящий
    после этого замолкает или заметно скомкан."""
    segs = sorted([s for s in transcript.segments if s.end > s.start], key=lambda s: s.start)
    events: List[Dict[str, Any]] = []
    overlap_total = 0.0
    counted = set()

    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            if b.start >= a.end:
                break
            if b.speaker == a.speaker:
                continue
            ov = min(a.end, b.end) - b.start
            if ov <= 0:
                continue
            overlap_total += ov
            key = (round(a.start, 2), round(b.start, 2))
            if key in counted:
                continue
            if ov < mcfg.interruption_overlap_sec:
                continue
            if b.word_count < mcfg.interruption_min_words:
                continue
            if _is_backchannel(b, mcfg):
                continue
            # перебивающий должен продолжить говорить после того, как первый замолк
            if b.end <= a.end + 0.2:
                continue
            counted.add(key)
            events.append({
                "by": b.speaker,
                "of": a.speaker,
                "at": round(b.start, 2),
                "overlap_sec": round(ov, 2),
                "quote": b.text.strip()[:160],
                "interrupted_quote": a.text.strip()[:160],
            })
    return events, round(overlap_total, 2)


def compute(transcript: Transcript, cfg: Config, silence_ratio: float = 0.0
            ) -> Dict[str, Any]:
    m = cfg.metrics
    op = transcript.by_speaker(OPERATOR)
    cl = transcript.by_speaker(CLIENT)
    dur = max(transcript.duration, 0.001)

    op_spans, cl_spans = _merge_spans(op), _merge_spans(cl)
    op_sec, cl_sec = _total(op_spans), _total(cl_spans)
    speech = op_sec + cl_sec

    events, overlap = detect_interruptions(transcript, m)
    op_int = sum(1 for e in events if e["by"] == OPERATOR)
    cl_int = sum(1 for e in events if e["by"] == CLIENT)

    # паузы между репликами любой из сторон
    all_spans = _merge_spans(transcript.segments)
    pauses: List[Tuple[float, float]] = []
    for (s1, e1), (s2, _e2) in zip(all_spans, all_spans[1:]):
        if s2 - e1 > 0:
            pauses.append((e1, s2 - e1))
    long_pauses = [p for p in pauses if p[1] >= m.long_pause_sec]

    # монологи: непрерывные серии реплик одной стороны
    monologues = _monologues(transcript)
    op_mono = [d for spk, d, _st in monologues if spk == OPERATOR and d >= m.monologue_sec]
    longest_mono = max([d for _s, d, _st in monologues] or [0.0])

    op_words = sum(s.word_count for s in op)
    cl_words = sum(s.word_count for s in cl)

    # стоп-фразы и мат в речи оператора
    op_idx = SpeechIndex(op, lemmatize_on=cfg.matching.lemmatize)
    stop_hits: List[Evidence] = []
    for ph in (cfg.filters.stop_phrases or []):
        ev = find_pattern(op_idx, ph, fuzzy_threshold=cfg.matching.fuzzy_threshold,
                          fuzzy_min_len=cfg.matching.fuzzy_min_len,
                          max_gap_words=cfg.matching.max_gap_words)
        if ev:
            stop_hits.append(ev)
    prof_hits: List[Evidence] = []
    any_idx = SpeechIndex(transcript.segments, lemmatize_on=cfg.matching.lemmatize)
    for ph in (cfg.filters.profanity or []):
        ev = find_pattern(any_idx, ph, fuzzy_threshold=0)
        if ev:
            prof_hits.append(ev)

    out: Dict[str, Any] = {
        "duration": round(dur, 2),
        "operator_talk_sec": round(op_sec, 2),
        "client_talk_sec": round(cl_sec, 2),
        "operator_talk_ratio": round(op_sec / speech, 3) if speech > 0 else 0.0,
        "client_talk_ratio": round(cl_sec / speech, 3) if speech > 0 else 0.0,
        "silence_ratio": round(silence_ratio if silence_ratio else
                               max(0.0, 1 - _total(all_spans) / dur), 3),
        "operator_interruptions": op_int,
        "client_interruptions": cl_int,
        "total_interruptions": len(events),
        "overlap_sec": overlap,
        "long_pauses": len(long_pauses),
        "longest_pause_sec": round(max([p[1] for p in pauses] or [0.0]), 2),
        "operator_monologues": len(op_mono),
        "longest_monologue_sec": round(longest_mono, 2),
        "operator_wpm": round(op_words / (op_sec / 60), 1) if op_sec > 1 else 0.0,
        "client_wpm": round(cl_words / (cl_sec / 60), 1) if cl_sec > 1 else 0.0,
        "operator_words": op_words,
        "client_words": cl_words,
        "turns": len(transcript.segments),
        "stop_phrase_hits": len(stop_hits),
        "profanity_hits": len(prof_hits),
        "client_questions": sum(1 for s in cl if "?" in s.text),
        "operator_questions": sum(1 for s in op if "?" in s.text),
        "response_delay_avg": _avg_response_delay(transcript),
        # служебное — не выводится как метрика, используется в отчёте
        "_interruption_events": events,
        "_long_pauses": [{"at": round(a, 2), "sec": round(b, 2)} for a, b in long_pauses],
        "_stop_phrase_evidence": stop_hits,
        "_profanity_evidence": prof_hits,
        "_monologues": [{"speaker": s, "sec": round(d, 1), "at": round(st, 1)}
                        for s, d, st in monologues if d >= m.monologue_sec],
    }
    return out


def _monologues(transcript: Transcript) -> List[Tuple[str, float, float]]:
    segs = sorted(transcript.segments, key=lambda s: s.start)
    out: List[Tuple[str, float, float]] = []
    cur_spk, start, end = None, 0.0, 0.0
    for s in segs:
        if s.speaker != cur_spk:
            if cur_spk is not None:
                out.append((cur_spk, end - start, start))
            cur_spk, start = s.speaker, s.start
        end = max(end, s.end)
    if cur_spk is not None:
        out.append((cur_spk, end - start, start))
    return out


def _avg_response_delay(transcript: Transcript) -> float:
    segs = sorted(transcript.segments, key=lambda s: s.start)
    delays = []
    for prev, cur in zip(segs, segs[1:]):
        if prev.speaker == CLIENT and cur.speaker == OPERATOR and cur.start > prev.end:
            delays.append(cur.start - prev.end)
    return round(sum(delays) / len(delays), 2) if delays else 0.0


OPS = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: abs(a - b) < 1e-9,
    "!=": lambda a, b: abs(a - b) >= 1e-9,
}
