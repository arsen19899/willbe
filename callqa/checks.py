"""Оценка чек-листа: ключевые фразы, регулярки, метрики, товары."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .config import Check, Config
from .metrics import METRIC_TITLES, OPS
from .models import (CLIENT, OPERATOR, CheckResult, Evidence, ProductResult,
                     Transcript)
from .text import (SpeechIndex, count_occurrences, find_occurrences,
                   find_pattern)


class Matcher:
    """Кэширует индексы речи по спикерам, чтобы не строить их на каждый пункт."""

    def __init__(self, transcript: Transcript, cfg: Config):
        self.cfg = cfg
        self.mt = cfg.matching
        self._cache: Dict[str, SpeechIndex] = {}
        self.transcript = transcript

    def index(self, speaker: str) -> SpeechIndex:
        if speaker not in self._cache:
            self._cache[speaker] = SpeechIndex(
                self.transcript.by_speaker(speaker),
                lemmatize_on=self.mt.lemmatize,
                yo_to_e=self.mt.yo_to_e,
            )
        return self._cache[speaker]

    def find(self, speaker: str, pattern: str,
             t_from: Optional[float] = None, t_to: Optional[float] = None
             ) -> Optional[Evidence]:
        idx = self.index(speaker)
        if t_from is not None or t_to is not None:
            idx = idx.slice_by_time(t_from, t_to)
        return find_pattern(
            idx, pattern,
            fuzzy_threshold=self.mt.fuzzy_threshold,
            fuzzy_min_len=self.mt.fuzzy_min_len,
            max_gap_words=self.mt.max_gap_words,
            evidence_chars=self.cfg.output.evidence_chars,
        )


# --------------------------------------------------------------------------- #
def evaluate_keyword(check: Check, m: Matcher) -> Tuple[Optional[bool], str, List[Evidence]]:
    ev: List[Evidence] = []
    spk = check.speaker

    # пустой пункт нельзя проверить — иначе он молча засчитался бы как «было»
    if not (check.any_of or check.all_of or check.none_of):
        return None, "фразы не заданы — заполните пункт в настройках", []

    any_ok = True
    hit_count = 0
    if check.any_of:
        if check.min_hits > 1:
            idx = m.index(spk)
            if check.time_from is not None or check.time_to is not None:
                idx = idx.slice_by_time(check.time_from, check.time_to)
            hit_count, found = count_occurrences(idx, check.any_of,
                                                 m.mt.max_gap_words)
            ev.extend(found[:5])
            any_ok = hit_count >= check.min_hits
        else:
            found = [m.find(spk, p, check.time_from, check.time_to) for p in check.any_of]
            found = [f for f in found if f]
            ev.extend(found)
            hit_count = len(found)
            any_ok = bool(found)

    all_ok = True
    missing_all: List[str] = []
    for p in check.all_of:
        f = m.find(spk, p, check.time_from, check.time_to)
        if f:
            ev.append(f)
        else:
            all_ok = False
            missing_all.append(p)

    none_hits: List[Evidence] = []
    for p in check.none_of:
        f = m.find(spk, p, check.time_from, check.time_to)
        if f:
            none_hits.append(f)
    ev.extend(none_hits)

    passed = any_ok and all_ok and not none_hits
    if check.invert:
        # инвертированный пункт: «пройден, если ничего не найдено»
        hit = bool([e for e in ev if e])
        passed = not hit
        detail = ("Запрещённые фразы не встречались" if passed
                  else "Найдено: " + "; ".join(f'«{e.pattern}» в {e.ts()}' for e in ev[:3]))
        return passed, detail, ev

    parts = []
    if check.any_of:
        if check.min_hits > 1:
            parts.append(f"найдено совпадений: {hit_count} (нужно ≥ {check.min_hits})")
        else:
            parts.append("совпадение найдено" if any_ok
                         else "ни одна из фраз не встретилась")
    if missing_all:
        parts.append("не хватает: " + ", ".join(f'«{p}»' for p in missing_all[:3]))
    if none_hits:
        parts.append("встретились запрещённые: " +
                     ", ".join(f'«{e.pattern}»' for e in none_hits[:3]))
    if check.time_to is not None and not any_ok:
        parts.append(f"искали в первые {int(check.time_to)} с")
    return passed, "; ".join(parts) or ("найдено" if passed else "не найдено"), ev


def evaluate_sequence(check: Check, m: Matcher
                      ) -> Tuple[Optional[bool], str, List[Evidence]]:
    """«Реплика одного — реакция другого в течение N секунд».

    Так проверяются вещи, которые раньше требовали языковой модели:
    отработку возражения, ответ на вопрос клиента, реакцию на недовольство.
    Если повода не было (клиент не возражал) — поведение задаёт if_no_trigger.
    """
    if not check.trigger_any_of or not check.any_of:
        return None, "не заданы реплики-поводы или фразы ответа", []

    trig_idx = m.index(check.trigger_speaker)
    triggers: List[Evidence] = []
    for p in check.trigger_any_of:
        triggers.extend(find_occurrences(trig_idx, p,
                                         max_gap_words=m.mt.max_gap_words))
    triggers.sort(key=lambda e: e.start)

    if not triggers:
        mapping = {"pass": (True, "повода не было — пункт не применяется"),
                   "fail": (False, "повод не найден"),
                   "na": (None, "повода не было")}
        passed, detail = mapping.get(check.if_no_trigger, (True, "повода не было"))
        return passed, detail, []

    resp_idx = m.index(check.speaker)
    for trig in triggers:
        window = resp_idx.slice_by_time(trig.start, trig.end + check.within_sec)
        for p in check.any_of:
            ev = find_pattern(window, p,
                              fuzzy_threshold=m.mt.fuzzy_threshold,
                              fuzzy_min_len=m.mt.fuzzy_min_len,
                              max_gap_words=m.mt.max_gap_words,
                              evidence_chars=m.cfg.output.evidence_chars)
            if ev:
                return True, (f"в {trig.ts()} прозвучало «{_said(trig)}» — "
                              f"в {ev.ts()} последовало «{_said(ev)}»"), [trig, ev]

    first = triggers[0]
    return False, (f"повод был ({len(triggers)} шт., первый в {first.ts()}: "
                   f"«{_said(first)}»), но реакции в течение "
                   f"{int(check.within_sec)} с не нашлось"), triggers[:3]


def _said(ev: Evidence) -> str:
    """Что реально прозвучало. Для регулярки её текст показывать бесполезно."""
    return (ev.matched or ev.pattern or "").strip()[:50]


def _num(v: float) -> str:
    """1.0 -> «1», 0.675 -> «0.675» — чтобы обоснование читалось по-человечески."""
    f = float(v)
    return str(int(f)) if abs(f - round(f)) < 1e-9 else str(round(f, 3))


def evaluate_metric(check: Check, metrics: Dict[str, Any]
                    ) -> Tuple[Optional[bool], str, List[Evidence]]:
    if check.metric not in metrics:
        return None, f"метрика '{check.metric}' не рассчитана", []
    val = metrics[check.metric]
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None, f"метрика '{check.metric}' не числовая", []
    fn = OPS.get(check.op)
    if fn is None:
        return None, f"неизвестный оператор '{check.op}'", []
    passed = fn(val, float(check.value))
    title = METRIC_TITLES.get(check.metric, check.metric)
    detail = f"{title}: {_num(val)} (порог {check.op} {_num(check.value)})"

    ev: List[Evidence] = []
    if not passed:
        if check.metric == "operator_interruptions":
            for e in metrics.get("_interruption_events", []):
                if e["by"] == OPERATOR:
                    ev.append(Evidence(f'перебил: «{e["quote"]}»', e["at"], e["at"],
                                       OPERATOR, check.metric,
                                       f'наложение {e["overlap_sec"]} с', "metric"))
        elif check.metric == "stop_phrase_hits":
            ev.extend(metrics.get("_stop_phrase_evidence", []))
        elif check.metric == "long_pauses":
            for p in metrics.get("_long_pauses", [])[:5]:
                ev.append(Evidence(f'пауза {p["sec"]} с', p["at"], p["at"],
                                   "any", check.metric,
                                   f'{p["sec"]} с', "metric"))
    return passed, detail, ev[:5]


def evaluate_products(cfg: Config, m: Matcher) -> List[ProductResult]:
    out: List[ProductResult] = []
    for p in cfg.products:
        patterns = [p.name] + list(p.aliases or [])
        ev: List[Evidence] = []
        for pat in patterns:
            f = m.find(OPERATOR, pat)
            if f:
                ev.append(f)
                break
        out.append(ProductResult(
            name=p.name, offered=bool(ev), required=p.required,
            group=p.group, how=(ev[0].how if ev else ""), evidence=ev,
        ))
    return out


def run_rule_checks(cfg: Config, transcript: Transcript, metrics: Dict[str, Any],
                    m: Matcher, products: List[ProductResult]) -> List[CheckResult]:
    """Все пункты, кроме llm-типа. Пункты llm заполняются позже в llm.py."""
    results: List[CheckResult] = []
    for c in cfg.checks:
        base = dict(id=c.id, title=c.title, category=c.category, type=c.type,
                    speaker=c.speaker, weight=c.weight, required=c.required,
                    invert=c.invert)
        if c.type in ("keyword", "regex"):
            passed, detail, ev = evaluate_keyword(c, m)
        elif c.type == "sequence":
            passed, detail, ev = evaluate_sequence(c, m)
        elif c.type == "metric":
            passed, detail, ev = evaluate_metric(c, metrics)
        elif c.type == "product":
            required = [p for p in products if p.required]
            pool = required or products
            missing = [p.name for p in pool if not p.offered]
            offered = [p.name for p in pool if p.offered]
            passed = not missing
            detail = ("Предложены: " + ", ".join(offered) if offered else "Ничего не предложено")
            if missing:
                detail += ("; не предложены: " + ", ".join(missing))
            ev = [e for p in products if p.offered for e in p.evidence][:5]
        elif c.type == "llm":
            passed, detail, ev = None, "ожидает смысловой оценки", []
        else:
            passed, detail, ev = None, f"неизвестный тип '{c.type}'", []
        results.append(CheckResult(passed=passed, detail=detail, evidence=ev, **base))
    return results


# --------------------------------------------------------------------------- #
def score(results: List[CheckResult]) -> Tuple[float, float, float, str, Dict[str, float]]:
    """Итоговый балл: взвешенная доля пройденных пунктов. Пункты «н/д» не учитываются."""
    got = 0.0
    total = 0.0
    by_cat: Dict[str, List[float]] = {}
    for r in results:
        if r.passed is None:
            continue
        w = max(0.0, float(r.weight))
        total += w
        val = w if r.passed else 0.0
        got += val
        by_cat.setdefault(r.category, [0.0, 0.0])
        by_cat[r.category][0] += val
        by_cat[r.category][1] += w
    pct = round(100 * got / total, 1) if total > 0 else 0.0
    cats = {k: round(100 * v[0] / v[1], 1) if v[1] else 0.0 for k, v in by_cat.items()}
    return round(got, 2), round(total, 2), pct, grade_of(pct), cats


def grade_of(pct: float) -> str:
    if pct >= 90:
        return "Отлично"
    if pct >= 75:
        return "Хорошо"
    if pct >= 60:
        return "Удовлетворительно"
    if pct >= 40:
        return "Слабо"
    return "Плохо"
