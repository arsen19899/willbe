"""Определение, кто из говорящих оператор, а кто клиент."""
from __future__ import annotations

from typing import Dict, List, Tuple

from .config import SpeakersConfig
from .models import CLIENT, OPERATOR, Segment, Transcript
from .text import SpeechIndex, find_pattern

# Признаки, которые почти всегда принадлежат оператору
_BUILTIN_OPERATOR = [
    "меня зовут", "чем могу помочь", "чем могу быть полезен", "оставайтесь на линии",
    "минуточку", "секундочку", "уточню информацию", "оформляю заявку",
    "спасибо за обращение", "компания", "служба поддержки", "горячая линия",
    "ваш звонок очень важен", "хорошего дня",
]
_BUILTIN_CLIENT = [
    "я хотел бы", "я хотела бы", "мне нужно", "а сколько стоит", "у меня вопрос",
    "мне не подходит", "я подумаю", "мне звонили", "я заказывал", "я купил",
]


def _score_role(index: SpeechIndex, markers: List[str]) -> int:
    hits = 0
    for m in markers:
        if find_pattern(index, m, fuzzy_threshold=0, max_gap_words=1):
            hits += 1
    return hits


def resolve_roles(transcript: Transcript, cfg: SpeakersConfig) -> Tuple[Transcript, List[str]]:
    """Приводит метки спикеров к operator/client. Возвращает транскрипт и предупреждения."""
    warnings: List[str] = []
    labels = sorted({s.speaker for s in transcript.segments})

    op_markers = (cfg.operator_markers or []) + _BUILTIN_OPERATOR
    cl_markers = (cfg.client_markers or []) + _BUILTIN_CLIENT

    # ---- случай стерео: роли уже назначены, но можно проверить ------------ #
    if set(labels) <= {OPERATOR, CLIENT} and labels:
        if cfg.autodetect_roles and len(labels) == 2:
            op_idx = SpeechIndex(transcript.by_speaker(OPERATOR))
            cl_idx = SpeechIndex(transcript.by_speaker(CLIENT))
            a = _score_role(op_idx, op_markers) - _score_role(op_idx, cl_markers)
            b = _score_role(cl_idx, op_markers) - _score_role(cl_idx, cl_markers)
            if b - a >= 2:      # уверенный перевес -> каналы перепутаны
                for s in transcript.segments:
                    s.speaker = CLIENT if s.speaker == OPERATOR else OPERATOR
                warnings.append(
                    "Дорожки поменяны местами: по репликам оператором оказался "
                    f"{'правый' if cfg.left == OPERATOR else 'левый'} канал."
                )
        return transcript, warnings

    if not labels:
        return transcript, ["Речь не распознана."]

    if len(labels) == 1:
        for s in transcript.segments:
            s.speaker = OPERATOR
        warnings.append("В записи найден только один голос — вся речь отнесена к оператору.")
        return transcript, warnings

    # ---- случай диаризации: считаем очки по каждому голосу ---------------- #
    scores: Dict[str, float] = {}
    for lab in labels:
        idx = SpeechIndex([s for s in transcript.segments if s.speaker == lab])
        score = _score_role(idx, op_markers) * 2.0 - _score_role(idx, cl_markers) * 2.0
        scores[lab] = score

    # тай-брейк: обычно первым говорит оператор (исходящий/приветствие)
    first = min(transcript.segments, key=lambda s: s.start).speaker
    scores[first] = scores.get(first, 0.0) + 0.5

    ranked = sorted(labels, key=lambda l: scores[l], reverse=True)
    op_label = ranked[0]
    # два самых «говорящих» голоса, остальные — к ближайшему по объёму
    talk: Dict[str, float] = {}
    for s in transcript.segments:
        talk[s.speaker] = talk.get(s.speaker, 0.0) + s.duration
    main = sorted(labels, key=lambda l: talk.get(l, 0.0), reverse=True)[:2]
    cl_label = next((l for l in main if l != op_label), None)
    if cl_label is None:
        cl_label = next((l for l in ranked if l != op_label), op_label)

    mapping = {op_label: OPERATOR, cl_label: CLIENT}
    extra = [l for l in labels if l not in mapping]
    if extra:
        warnings.append(
            f"В записи найдено голосов: {len(labels)}. Дополнительные "
            f"({', '.join(extra)}) отнесены к клиенту."
        )
    for lab in extra:
        mapping[lab] = CLIENT

    for s in transcript.segments:
        s.speaker = mapping.get(s.speaker, CLIENT)

    if abs(scores[op_label] - scores.get(cl_label, 0.0)) < 1.0:
        warnings.append(
            "Роли оператор/клиент определены с низкой уверенностью — "
            "проверьте расшифровку выборочно."
        )
    return transcript, warnings
