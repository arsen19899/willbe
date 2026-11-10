"""Обучение правил на размеченных записях.

Вход: две папки — успешные разговоры и неуспешные. Программа расшифровывает
их, ищет фразы и метрики, которые статистически отличают одни от других,
и предлагает готовые пункты чек-листа.

Ключевые принципы, без которых такой поиск выдаёт мусор:

  * сравниваются ДОЛИ ЗВОНКОВ с фразой, а не количество употреблений —
    иначе один болтливый разговор перевешивает десять коротких;
  * значимость считается точным тестом Фишера, а не «на глаз»;
  * применяется поправка на множественные сравнения: перебирая тысячи фраз,
    без неё гарантированно наберёшь случайных «правил»;
  * результат проверяется на отложенных данных — если правила не работают
    на звонках, которых не видели при обучении, программа об этом скажет;
  * фразы, звучащие в самом конце разговора, помечаются: обычно это
    следствие успеха («оформляем заказ»), а не его причина.

Ничего не перезаписывается автоматически: на выходе — предложение,
которое человек принимает по пунктам и может править руками.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import audio as au, metrics as mt, stats
from .config import Config
from .models import CLIENT, OPERATOR, Segment, Transcript, Word
from .text import SpeechIndex, lemma, tokenize

GOOD = "good"
BAD = "bad"

# Служебные слова: сами по себе ничего не значат, но внутри фразы полезны
STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их",
    "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою",
    "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между", "это", "ага", "угу",
    "э", "эм", "мм", "аа", "вообще", "просто", "как-то", "типа",
}

CACHE_DIR = "cache"


# --------------------------------------------------------------------------- #
#  Данные
# --------------------------------------------------------------------------- #
@dataclass
class LabeledCall:
    path: str
    label: str                       # good | bad
    transcript: Optional[Transcript] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class PhraseCandidate:
    phrase: str                      # как писать в правило
    display: str                     # как показать человеку
    speaker: str                     # operator | client
    direction: str                   # good | bad — где встречается чаще
    good_calls: int
    bad_calls: int
    good_total: int
    bad_total: int
    p_value: float
    q_value: float                   # с поправкой на множественные сравнения
    lift: float                      # во сколько раз чаще
    mean_position: float             # 0 = начало разговора, 1 = конец
    outcome_risk: bool               # похоже на следствие успеха, а не причину
    examples: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)   # другие формы той же находки
    _good_set: frozenset = field(default=frozenset(), repr=False, compare=False)
    _bad_set: frozenset = field(default=frozenset(), repr=False, compare=False)

    @property
    def good_rate(self) -> float:
        return self.good_calls / self.good_total if self.good_total else 0.0

    @property
    def bad_rate(self) -> float:
        return self.bad_calls / self.bad_total if self.bad_total else 0.0


@dataclass
class MetricCandidate:
    metric: str
    title: str
    op: str
    threshold: float
    good_mean: float
    bad_mean: float
    auc: float
    accuracy: float
    p_value: float
    q_value: float
    direction: str


@dataclass
class LearnResult:
    good_total: int = 0
    bad_total: int = 0
    failed: List[str] = field(default_factory=list)
    phrases: List[PhraseCandidate] = field(default_factory=list)
    metrics: List[MetricCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for p, src in zip(d["phrases"], self.phrases):
            p["good_rate"] = round(src.good_rate, 3)
            p["bad_rate"] = round(src.bad_rate, 3)
        return d


# --------------------------------------------------------------------------- #
#  Расшифровка с кэшем
# --------------------------------------------------------------------------- #
def _file_key(path: str) -> str:
    st = os.stat(path)
    h = hashlib.sha1()
    h.update(str(st.st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:20]


def _cache_path(cache_dir: str, path: str, model: str) -> str:
    return os.path.join(cache_dir, f"{_file_key(path)}_{model.replace('/', '_')}.json")


def _save_cache(p: str, tr: Transcript, metrics: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    data = {
        "duration": tr.duration, "channels": tr.channels, "source": tr.source,
        "segments": [{"speaker": s.speaker, "start": s.start, "end": s.end,
                      "text": s.text,
                      "words": [{"text": w.text, "start": w.start, "end": w.end}
                                for w in s.words]} for s in tr.segments],
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_cache(p: str) -> Optional[Tuple[Transcript, Dict[str, Any]]]:
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    segs = [Segment(s["speaker"], s["start"], s["end"], s["text"],
                    [Word(w["text"], w["start"], w["end"]) for w in s.get("words", [])])
            for s in d.get("segments", [])]
    tr = Transcript(segments=segs, duration=d.get("duration", 0.0),
                    channels=d.get("channels", 1), source=d.get("source", ""))
    return tr, d.get("metrics", {})


def load_corpus(good_dir: str, bad_dir: str, cfg: Config,
                progress: Optional[Callable[[str, float], None]] = None,
                cache_dir: Optional[str] = None,
                stop_flag: Optional[Callable[[], bool]] = None) -> List[LabeledCall]:
    """Расшифровывает обе папки. Готовые расшифровки берутся из кэша —
    повторный прогон обучения не тратит время на распознавание заново."""
    from .pipeline import analyze_file

    cache_dir = cache_dir or os.path.join(cfg.output.dir, CACHE_DIR)
    items: List[LabeledCall] = []
    for folder, label in ((good_dir, GOOD), (bad_dir, BAD)):
        if not folder:
            continue
        for p in au.list_audio_files(folder):
            items.append(LabeledCall(path=p, label=label))

    total = max(1, len(items))
    for i, call in enumerate(items):
        if stop_flag and stop_flag():
            break
        if progress:
            tag = "успешные" if call.label == GOOD else "неуспешные"
            progress(f"[{i + 1}/{total}] {tag}: {call.name}", i / total)

        cached = None
        try:
            cpath = _cache_path(cache_dir, call.path, cfg.asr.model)
            cached = _load_cache(cpath)
        except OSError:
            cpath = None

        if cached:
            call.transcript, call.metrics = cached
            continue

        res = analyze_file(call.path, cfg)
        if not res.ok or not res.transcript:
            call.ok = False
            call.error = res.skipped_reason or res.error or "не разобран"
            continue
        call.transcript = res.transcript
        call.metrics = res.metrics
        if cpath:
            try:
                _save_cache(cpath, res.transcript, res.metrics)
            except OSError:
                pass

    if progress:
        progress("расшифровка готова", 1.0)
    return items


# --------------------------------------------------------------------------- #
#  Извлечение фраз
# --------------------------------------------------------------------------- #
def call_ngrams(transcript: Transcript, speaker: str, max_n: int = 4
                ) -> Dict[str, Tuple[str, float]]:
    """Все n-граммы речи одной стороны -> (пример в исходном виде, доля позиции).

    Возвращается именно множество (по одному разу на звонок): дальше
    сравниваются доли звонков, а не частоты, чтобы длинный разговор
    не перевешивал короткий.
    """
    idx = SpeechIndex(transcript.by_speaker(speaker))
    n_toks = len(idx.toks)
    if n_toks == 0:
        return {}
    dur = max(transcript.duration, 0.001)

    out: Dict[str, Tuple[str, float]] = {}
    for n in range(1, max_n + 1):
        for i in range(n_toks - n + 1):
            window = idx.toks[i:i + n]
            # фраза не должна склеивать конец одной реплики с началом следующей:
            # так рождаются химеры вроде «чем могу помочь честно»
            if window[0].seg != window[-1].seg:
                continue
            lemmas = [t.lem for t in window]
            if all(l in STOPWORDS for l in lemmas):
                continue
            if n == 1 and (lemmas[0] in STOPWORDS or len(lemmas[0]) < 3
                           or lemmas[0].isdigit()):
                continue
            key = " ".join(lemmas)
            if key in out:
                continue
            surface = " ".join(t.raw for t in window)
            out[key] = (surface, min(1.0, window[0].start / dur))
    return out


def mine_phrases(calls: Sequence[LabeledCall], min_calls: int = 3,
                 max_q: float = 0.10, min_lift: float = 1.6,
                 max_n: int = 4, speakers: Sequence[str] = (OPERATOR, CLIENT),
                 late_threshold: float = 0.8) -> List[PhraseCandidate]:
    """Ищет фразы, которые статистически отличают успешные звонки от неуспешных."""
    usable = [c for c in calls if c.ok and c.transcript]
    good = [c for c in usable if c.label == GOOD]
    bad = [c for c in usable if c.label == BAD]
    if not good or not bad:
        return []

    # для каждой фразы: в скольких звонках каждой группы встретилась
    counts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for call in usable:
        for spk in speakers:
            grams = call_ngrams(call.transcript, spk, max_n)
            for key, (surface, pos) in grams.items():
                slot = counts.setdefault((spk, key), {
                    "good": 0, "bad": 0, "surface": {}, "pos": [], "examples": [],
                    "good_set": set(), "bad_set": set()})
                slot[call.label] += 1
                slot[f"{call.label}_set"].add(call.path)
                slot["surface"][surface.lower()] = \
                    slot["surface"].get(surface.lower(), 0) + 1
                slot["pos"].append(pos)
                if len(slot["examples"]) < 3:
                    slot["examples"].append(f"{call.name}: …{surface}…")

    ng, nb = len(good), len(bad)
    raw: List[PhraseCandidate] = []
    for (spk, key), slot in counts.items():
        a, b = slot["good"], slot["bad"]
        if a + b < min_calls:
            continue
        rate_g = a / ng
        rate_b = b / nb
        # сглаживание: без него одно вхождение против нуля даёт бесконечный «эффект»
        lift = ((rate_g + 0.5 / ng) / (rate_b + 0.5 / nb))
        direction = GOOD if rate_g > rate_b else BAD
        effect = lift if direction == GOOD else 1 / lift
        if effect < min_lift:
            continue

        p = stats.fisher_exact(a, b, ng - a, nb - b)
        surface = max(slot["surface"].items(), key=lambda kv: kv[1])[0]
        mean_pos = stats.mean(slot["pos"])
        raw.append(PhraseCandidate(
            phrase=surface, display=surface, speaker=spk, direction=direction,
            good_calls=a, bad_calls=b, good_total=ng, bad_total=nb,
            p_value=p, q_value=1.0, lift=round(effect, 2),
            mean_position=round(mean_pos, 3),
            outcome_risk=(direction == GOOD and mean_pos >= late_threshold),
            examples=slot["examples"],
            _good_set=frozenset(slot["good_set"]),
            _bad_set=frozenset(slot["bad_set"]),
        ))

    if not raw:
        return []

    qs = stats.benjamini_hochberg([c.p_value for c in raw])
    for c, q in zip(raw, qs):
        c.q_value = q

    kept = [c for c in raw if c.q_value <= max_q]
    kept = _drop_redundant(kept)
    kept.sort(key=lambda c: (c.q_value, -c.lift))
    return kept


# Части речи, на которых фраза не должна обрываться или начинаться:
# «оформить расширенную» и «и потому что» — не формулировки, а обрубки
_BAD_TAIL = {"ADJF", "ADJS", "PRTF", "PREP", "CONJ", "PRCL", "NPRO", "Apro"}
_BAD_HEAD = {"PREP", "CONJ", "PRCL"}


def _pos(word: str) -> str:
    """Часть речи слова; пустая строка, если морфология недоступна."""
    from .text import _morph

    m = _morph()
    if m is None:
        return ""
    try:
        return str(m.parse(word)[0].tag.POS or "")
    except Exception:
        return ""


def _dangling(toks: Sequence[str]) -> bool:
    """Фраза обрывается на полуслове? «оформить расширенную» — да."""
    if not toks:
        return True
    return _pos(toks[-1]) in _BAD_TAIL or _pos(toks[0]) in _BAD_HEAD


def _representative_score(c: PhraseCandidate) -> tuple:
    """Какую из равнозначных формулировок показать человеку.

    Слишком длинная фраза хрупкая (не сработает при малейшем отличии),
    слишком короткая срабатывает где попало. Оптимум — два-три слова,
    и фраза не должна обрываться на прилагательном или предлоге.
    """
    toks = c.phrase.lower().split()
    content = sum(1 for t in toks if lemma(t) not in STOPWORDS)
    length_penalty = max(0, len(toks) - 3)
    return (-int(_dangling(toks)), -length_penalty, len(toks), content, c.phrase)


def _drop_redundant(cands: List[PhraseCandidate]) -> List[PhraseCandidate]:
    """Схлопывает разные формулировки одной и той же находки.

    Скользящие окна по одному предложению — «для каких целей вам»,
    «каких целей вам нужна», «целей вам нужна лампа» — это не три находки,
    а одна. Признак: фразы встретились ровно в одних и тех же записях.
    Остаётся одна формулировка, остальные попадают в её variants,
    чтобы человек видел, из чего выбирали.
    """
    buckets: Dict[tuple, List[PhraseCandidate]] = {}
    for c in cands:
        key = (c.speaker, c.direction, c._good_set, c._bad_set)
        buckets.setdefault(key, []).append(c)

    # Одинаковый набор записей — необходимое условие, но не достаточное:
    # если оператор работает по скрипту, все его фразы встречаются во всех
    # звонках сразу. Поэтому дополнительно требуем пересечения по словам —
    # так объединяются только окна по одному и тому же предложению.
    groups: List[List[PhraseCandidate]] = []
    for bucket in buckets.values():
        clusters: List[Tuple[set, List[PhraseCandidate]]] = []
        for c in sorted(bucket, key=lambda x: -len(x.phrase.split())):
            words = set(c.phrase.lower().split())
            hit = next((cl for cl in clusters if cl[0] & words), None)
            if hit:
                hit[0].update(words)
                hit[1].append(c)
            else:
                clusters.append((set(words), [c]))
        groups.extend(cl[1] for cl in clusters)

    kept: List[PhraseCandidate] = []
    for group in groups:
        best = max(group, key=_representative_score)
        others = sorted((g.phrase for g in group if g is not best),
                        key=lambda p: (-len(p.split()), p))
        best.variants = others[:8]
        kept.append(best)

    # добить случаи, где наборы записей почти совпадают, но не буквально
    kept.sort(key=lambda c: (-len(c.phrase.split()), c.q_value))
    final: List[PhraseCandidate] = []
    for c in kept:
        words = set(c.phrase.lower().split())
        dup = False
        for k in final:
            if k.speaker != c.speaker or k.direction != c.direction:
                continue
            kw = set(k.phrase.lower().split())
            if (words <= kw or kw <= words) and \
                    k.good_calls == c.good_calls and k.bad_calls == c.bad_calls:
                if c.phrase not in k.variants:
                    k.variants = (k.variants + [c.phrase])[:8]
                dup = True
                break
        if not dup:
            final.append(c)
    return final


# --------------------------------------------------------------------------- #
#  Метрики
# --------------------------------------------------------------------------- #
SKIP_METRICS = {"duration", "operator_words", "client_words", "turns",
                "operator_talk_sec", "client_talk_sec"}


def mine_metrics(calls: Sequence[LabeledCall], max_q: float = 0.10,
                 min_auc: float = 0.65, min_rel_diff: float = 0.05
                 ) -> List[MetricCandidate]:
    usable = [c for c in calls if c.ok and c.metrics]
    good = [c for c in usable if c.label == GOOD]
    bad = [c for c in usable if c.label == BAD]
    if len(good) < 2 or len(bad) < 2:
        return []

    keys = [k for k in mt.METRIC_TITLES
            if k not in SKIP_METRICS
            and any(isinstance(c.metrics.get(k), (int, float)) for c in usable)]

    raw: List[MetricCandidate] = []
    for key in keys:
        gv = [float(c.metrics[key]) for c in good if isinstance(c.metrics.get(key), (int, float))]
        bv = [float(c.metrics[key]) for c in bad if isinstance(c.metrics.get(key), (int, float))]
        if len(gv) < 2 or len(bv) < 2:
            continue
        if len(set(gv + bv)) < 2:
            continue

        a = stats.auc(gv, bv)
        strength = max(a, 1 - a)
        if strength < min_auc:
            continue

        # Практическая значимость, а не только статистическая. «Темп речи
        # 156.0 против 156.1» может разделять группы идеально, но такое
        # правило бессмысленно и на новых записях развалится.
        gm, bm = stats.mean(gv), stats.mean(bv)
        scale = max(abs(gm), abs(bm), 1e-9)
        if abs(gm - bm) / scale < min_rel_diff:
            continue
        cut, op, acc = stats.best_threshold(gv, bv)

        # значимость самого разделения — по таблице «прошёл порог / не прошёл»
        passed_g = sum(1 for v in gv if (v <= cut if op == "<=" else v >= cut))
        passed_b = sum(1 for v in bv if (v <= cut if op == "<=" else v >= cut))
        p = stats.fisher_exact(passed_g, passed_b,
                               len(gv) - passed_g, len(bv) - passed_b)
        raw.append(MetricCandidate(
            metric=key, title=mt.METRIC_TITLES.get(key, key), op=op,
            threshold=round(cut, 3), good_mean=round(stats.mean(gv), 3),
            bad_mean=round(stats.mean(bv), 3), auc=round(a, 3),
            accuracy=round(acc, 3), p_value=p, q_value=1.0,
            direction=GOOD if a > 0.5 else BAD,
        ))

    if not raw:
        return []
    qs = stats.benjamini_hochberg([c.p_value for c in raw])
    for c, q in zip(raw, qs):
        c.q_value = q
    kept = [c for c in raw if c.q_value <= max_q]
    kept.sort(key=lambda c: (-c.accuracy, c.q_value))
    return kept


# --------------------------------------------------------------------------- #
#  Проверка на отложенных данных
# --------------------------------------------------------------------------- #
def cross_validate(calls: Sequence[LabeledCall], folds: int = 5,
                   **mine_kw) -> Dict[str, Any]:
    """Честная оценка: правила ищутся на части звонков, проверяются на остальных.

    Без этого шага нельзя отличить настоящую закономерность от подгонки
    под конкретные записи. Если точность близка к 50%, найденные правила
    не работают за пределами обучающей выборки.
    """
    usable = [c for c in calls if c.ok and c.transcript]
    good = [c for c in usable if c.label == GOOD]
    bad = [c for c in usable if c.label == BAD]
    if len(good) < folds or len(bad) < folds:
        return {"ok": False,
                "reason": f"для проверки нужно минимум по {folds} записи в каждой папке"}

    correct = total = 0
    scores_g: List[float] = []
    scores_b: List[float] = []
    for f in range(folds):
        test = good[f::folds] + bad[f::folds]
        train = [c for c in usable if c not in test]
        if not any(c.label == GOOD for c in train) or \
           not any(c.label == BAD for c in train):
            continue
        rules = mine_phrases(train, **mine_kw)
        if not rules:
            continue
        for call in test:
            score = _score_call(call, rules)
            (scores_g if call.label == GOOD else scores_b).append(score)
            predicted = GOOD if score > 0 else BAD
            correct += int(predicted == call.label)
            total += 1

    if total == 0:
        return {"ok": False, "reason": "правил не нашлось — проверять нечего"}

    return {
        "ok": True,
        "folds": folds,
        "checked": total,
        "accuracy": round(correct / total, 3),
        "auc": round(stats.auc(scores_g, scores_b), 3),
        "baseline": round(max(len(good), len(bad)) / len(usable), 3),
    }


def _score_call(call: LabeledCall, rules: Sequence[PhraseCandidate]) -> float:
    """Плюс за признак успешного разговора, минус за признак неуспешного."""
    if not call.transcript:
        return 0.0
    score = 0.0
    cache: Dict[str, Dict[str, Tuple[str, float]]] = {}
    for r in rules:
        if r.speaker not in cache:
            cache[r.speaker] = call_ngrams(call.transcript, r.speaker)
        key = " ".join(lemma(t) for t in tokenize(r.phrase))
        if key in cache[r.speaker]:
            score += 1.0 if r.direction == GOOD else -1.0
    return score


# --------------------------------------------------------------------------- #
#  Сборка предложения
# --------------------------------------------------------------------------- #
def analyse(good_dir: str, bad_dir: str, cfg: Config,
            progress: Optional[Callable[[str, float], None]] = None,
            min_calls: int = 3, max_q: float = 0.10, min_lift: float = 1.6,
            max_n: int = 4, calls: Optional[Sequence[LabeledCall]] = None,
            stop_flag: Optional[Callable[[], bool]] = None) -> LearnResult:
    if calls is None:
        calls = load_corpus(good_dir, bad_dir, cfg, progress, stop_flag=stop_flag)

    usable = [c for c in calls if c.ok and c.transcript]
    good = [c for c in usable if c.label == GOOD]
    bad = [c for c in usable if c.label == BAD]

    res = LearnResult(
        good_total=len(good), bad_total=len(bad),
        failed=[f"{c.name}: {c.error}" for c in calls if not c.ok],
        settings={"min_calls": min_calls, "max_q": max_q, "min_lift": min_lift,
                  "max_n": max_n},
    )

    if not good or not bad:
        res.warnings.append(
            "Нужны записи в обеих папках: успешные и неуспешные. "
            f"Сейчас успешных {len(good)}, неуспешных {len(bad)}.")
        return res

    res.warnings.extend(_sample_warnings(good, bad))

    if progress:
        progress("ищу отличающие фразы", 0.9)
    res.phrases = mine_phrases(usable, min_calls=min_calls, max_q=max_q,
                               min_lift=min_lift, max_n=max_n)
    res.metrics = mine_metrics(usable, max_q=max_q)

    if progress:
        progress("проверяю на отложенных записях", 0.95)
    res.validation = cross_validate(usable, min_calls=min_calls, max_q=max_q,
                                    min_lift=min_lift, max_n=max_n)

    res.warnings.extend(_result_warnings(res))
    if progress:
        progress("готово", 1.0)
    return res


def _sample_warnings(good: Sequence[LabeledCall],
                     bad: Sequence[LabeledCall]) -> List[str]:
    out: List[str] = []
    n = min(len(good), len(bad))
    if n < 8:
        out.append(
            f"Записей очень мало (успешных {len(good)}, неуспешных {len(bad)}). "
            "На такой выборке почти любое совпадение случайно. "
            "Отнеситесь к находкам как к подсказкам для ручной проверки, "
            "а не как к готовым правилам. Надёжный минимум — по 20–30 записей.")
    elif n < 20:
        out.append(
            f"Выборка небольшая (успешных {len(good)}, неуспешных {len(bad)}). "
            "Часть находок может оказаться случайной — сверяйтесь с цитатами.")

    ratio = len(good) / len(bad) if bad else 0
    if ratio and (ratio > 3 or ratio < 1 / 3):
        out.append(
            f"Папки сильно разного размера ({len(good)} против {len(bad)}). "
            "Выводы смещаются в сторону большей — по возможности выровняйте.")

    # длительность как скрытый фактор
    gd = [c.metrics.get("duration", 0) for c in good if c.metrics]
    bd = [c.metrics.get("duration", 0) for c in bad if c.metrics]
    if gd and bd:
        a = stats.auc(gd, bd)
        if max(a, 1 - a) > 0.75:
            longer = "успешные" if a > 0.5 else "неуспешные"
            out.append(
                f"Заметный перекос по длительности: {longer} разговоры почти всегда "
                f"длиннее (AUC {max(a, 1 - a):.2f}). Многие «отличающие» фразы могут "
                "объясняться просто тем, что в длинном разговоре успевает прозвучать "
                "больше слов. Проверяйте находки на осмысленность.")
    return out


def _result_warnings(res: LearnResult) -> List[str]:
    out: List[str] = []
    v = res.validation
    if v.get("ok"):
        acc, base = v["accuracy"], v["baseline"]
        if acc <= base + 0.05:
            out.append(
                f"На отложенных записях правила почти не работают: точность {acc:.0%} "
                f"против {base:.0%} у простого угадывания большей группы. "
                "Скорее всего, устойчивого различия в этих данных нет — "
                "добавьте записей или проверьте разметку папок.")
        elif acc < 0.7:
            out.append(
                f"Точность на отложенных записях умеренная ({acc:.0%}). "
                "Правила стоит использовать как ориентир, а не как приговор.")
    elif v.get("reason"):
        out.append("Проверку на отложенных записях выполнить не удалось: "
                   + v["reason"])

    risky = [p.display for p in res.phrases if p.outcome_risk]
    if risky:
        out.append(
            "Эти фразы звучат в самом конце разговора и, вероятно, являются "
            "следствием успеха, а не его причиной — как правило качества они "
            f"бесполезны: {', '.join(risky[:5])}. "
            "Пункт «оператор сказал “оформляем заказ”» не улучшит работу, "
            "если заказ и так состоялся.")
    if not res.phrases and not res.metrics:
        out.append(
            "Значимых отличий не найдено. Это не поломка: возможно, разница "
            "между группами не в словах, а в товаре, цене или клиенте. "
            "Можно ослабить требования (порог значимости и минимальный эффект) "
            "и посмотреть находки как гипотезы.")
    return out


# --------------------------------------------------------------------------- #
#  Превращение находок в пункты чек-листа
# --------------------------------------------------------------------------- #
def _slug(text: str, prefix: str) -> str:
    base = re.sub(r"[^a-zа-я0-9]+", "_", text.lower()).strip("_")[:28]
    return f"{prefix}_{base}" or prefix


def phrase_to_check(c: PhraseCandidate, category: str = "Найдено обучением"
                    ) -> Dict[str, Any]:
    """Признак успеха -> пункт «должно прозвучать».
    Признак неудачи -> пункт «не должно прозвучать» (invert)."""
    good_side = c.direction == GOOD
    title = (f"Прозвучало: «{c.display}»" if good_side
             else f"Не прозвучало: «{c.display}»")
    weight = round(min(10.0, max(2.0, (c.lift - 1) * 4)), 1)
    return {
        "id": _slug(c.display, "learn_pos" if good_side else "learn_neg"),
        "title": title,
        "category": category,
        "type": "keyword",
        "speaker": c.speaker,
        "weight": weight,
        "required": False,
        "invert": not good_side,
        "any_of": [c.display],
        "comment": (f"Обучение: встречается в {c.good_rate:.0%} успешных и "
                    f"{c.bad_rate:.0%} неуспешных звонков "
                    f"(в {c.lift:.1f}× чаще, q={c.q_value:.3f})"
                    + (f". Та же находка звучала и как: "
                       + ", ".join(f'«{v}»' for v in c.variants[:4])
                       if c.variants else "")),
    }


def metric_to_check(c: MetricCandidate, category: str = "Найдено обучением"
                    ) -> Dict[str, Any]:
    return {
        "id": _slug(c.metric, "learn_metric"),
        "title": f"{c.title} {c.op} {c.threshold:g}",
        "category": category,
        "type": "metric",
        "speaker": "any",
        "weight": round(min(10.0, max(2.0, (c.accuracy - 0.5) * 20)), 1),
        "required": False,
        "metric": c.metric,
        "op": c.op,
        "value": c.threshold,
        "comment": (f"Обучение: у успешных в среднем {c.good_mean:g}, "
                    f"у неуспешных {c.bad_mean:g}; "
                    f"разделяет с точностью {c.accuracy:.0%}, AUC {c.auc:g}"),
    }


def build_checks(res: LearnResult, phrase_ids: Optional[Sequence[int]] = None,
                 metric_ids: Optional[Sequence[int]] = None,
                 skip_outcome_risk: bool = True) -> List[Dict[str, Any]]:
    """Собирает пункты из выбранных находок. Ничего не выбрано — берём все,
    кроме помеченных как вероятное следствие успеха."""
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(res.phrases):
        if phrase_ids is not None and i not in phrase_ids:
            continue
        if phrase_ids is None and skip_outcome_risk and p.outcome_risk:
            continue
        out.append(phrase_to_check(p))
    for i, m in enumerate(res.metrics):
        if metric_ids is not None and i not in metric_ids:
            continue
        out.append(metric_to_check(m))

    seen: Dict[str, int] = {}
    for c in out:                     # id должны быть уникальны
        base = c["id"]
        if base in seen:
            seen[base] += 1
            c["id"] = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
    return out


def merge_into_config(cfg: Config, checks: List[Dict[str, Any]]) -> Config:
    """Добавляет пункты к существующему чек-листу, не трогая старые."""
    from .config import Config as C

    data = cfg.to_dict()
    existing = {c["id"] for c in data["checks"]}
    for c in checks:
        cid = c["id"]
        n = 1
        while cid in existing:
            n += 1
            cid = f"{c['id']}_{n}"
        c = dict(c, id=cid)
        existing.add(cid)
        data["checks"].append(c)
    return C.from_dict(data)
