"""Финальный итог по звонку и компактные формулировки для таблиц.

Вместо «было / не было» отчёт показывает, ЧТО именно сработало и КОГДА:
«меня зовут» 0:07. Итог собирается по правилам, языковая модель не нужна.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import Config
from .metrics import METRIC_TITLES
from .models import CLIENT, OPERATOR, ROLE_RU, CallResult, CheckResult, _ts

NOT_FOUND = "— не найдено"
NA = "н/д"


# --------------------------------------------------------------------------- #
#  Ячейка сводной таблицы
# --------------------------------------------------------------------------- #
def cell(c: CheckResult) -> str:
    """Короткая формулировка результата пункта: что нашли и на какой секунде."""
    if c.passed is None:
        return NA

    if c.type == "metric":
        return c.detail.split(" (порог")[0] if "(порог" in c.detail else c.detail

    if c.type == "sequence":
        if not c.passed:
            return NOT_FOUND
        if len(c.evidence) >= 2:
            t, r = c.evidence[0], c.evidence[1]
            return f"«{_short(t)}» {t.ts()} → «{_short(r)}» {r.ts()}"
        return "выполнено"

    if c.type == "product":
        return c.detail

    if not c.passed:
        if c.invert:
            return "чисто"
        return NOT_FOUND

    if c.invert:                     # пункт провален: нашли запрещённое
        return "чисто"

    if not c.evidence:
        return "выполнено"

    first = c.evidence[0]
    text = f"«{_short(first)}» {first.ts()}"
    if len(c.evidence) > 1:
        text += f" +{len(c.evidence) - 1}"
    return text


def status_of(c: CheckResult) -> str:
    if c.passed is None:
        return "na"
    return "yes" if c.passed else "no"


def _short(ev, limit: int = 30) -> str:
    """Что показать в таблице.

    Для регулярки сам шаблон читать невозможно, поэтому показываем то,
    что реально совпало в записи: «4500 рублей», а не «по шаблону».
    """
    pattern = (getattr(ev, "pattern", "") or "").strip()
    matched = (getattr(ev, "matched", "") or "").strip()
    p = matched if (pattern.lower().startswith("re:") and matched) else pattern
    if not p:
        p = matched or "совпадение"
    return p if len(p) <= limit else p[: limit - 1] + "…"


def product_cell(p) -> str:
    if not p.offered:
        return NOT_FOUND
    if p.evidence:
        e = p.evidence[0]
        how = " (модель)" if p.how == "llm" else ""
        return f"«{_short(e)}» {e.ts()}{how}"
    return "предложен"


# --------------------------------------------------------------------------- #
#  Все находки одним списком
# --------------------------------------------------------------------------- #
def findings(res: CallResult) -> List[Dict[str, Any]]:
    """Плоский журнал: каждое срабатывание с секундой, ролью и цитатой."""
    out: List[Dict[str, Any]] = []

    for c in res.checks:
        for e in c.evidence:
            out.append({
                "sec": round(e.start, 1),
                "time": e.ts(),
                "kind": "пункт чек-листа",
                "what": c.title,
                "pattern": _short(e, 60),
                "searched": e.pattern,
                "speaker": ROLE_RU.get(e.speaker or c.speaker, c.speaker),
                "quote": e.quote,
                "how": _how_ru(e.how),
                "result": c.status_ru,
            })

    for p in res.products:
        for e in p.evidence:
            out.append({
                "sec": round(e.start, 1),
                "time": e.ts(),
                "kind": "товар",
                "what": p.name,
                "pattern": _short(e, 60),
                "searched": e.pattern,
                "speaker": ROLE_RU.get(OPERATOR, OPERATOR),
                "quote": e.quote,
                "how": _how_ru(p.how or e.how),
                "result": "предложен",
            })

    for e in res.metrics.get("_interruption_events", []):
        out.append({
            "sec": round(e["at"], 1),
            "time": _ts(e["at"]),
            "kind": "перебивание",
            "what": f'{ROLE_RU.get(e["by"], e["by"])} перебил '
                    f'{ROLE_RU.get(e["of"], e["of"]).lower()}а',
            "pattern": f'наложение {e["overlap_sec"]} с',
            "searched": "",
            "speaker": ROLE_RU.get(e["by"], e["by"]),
            "quote": e["quote"],
            "how": "по таймкодам",
            "result": "",
        })

    for p in res.metrics.get("_long_pauses", []):
        out.append({
            "sec": round(p["at"], 1),
            "time": _ts(p["at"]),
            "kind": "пауза",
            "what": f'тишина {p["sec"]} с',
            "pattern": "",
            "searched": "",
            "speaker": "",
            "quote": "",
            "how": "по таймкодам",
            "result": "",
        })

    for e in res.metrics.get("_stop_phrase_evidence", []):
        out.append({
            "sec": round(e.start, 1),
            "time": e.ts(),
            "kind": "стоп-фраза",
            "what": e.pattern,
            "pattern": _short(e, 60),
            "searched": e.pattern,
            "speaker": ROLE_RU.get(OPERATOR, OPERATOR),
            "quote": e.quote,
            "how": _how_ru(e.how),
            "result": "нарушение",
        })

    out.sort(key=lambda x: x["sec"])
    return out


def _how_ru(how: str) -> str:
    return {
        "exact": "дословно",
        "lemma": "по форме слова",
        "fuzzy": "с поправкой на распознавание",
        "regex": "по шаблону",
        "metric": "по метрике",
        "llm": "смысловая оценка",
    }.get(how or "", how or "")


# --------------------------------------------------------------------------- #
#  Финальный итог
# --------------------------------------------------------------------------- #
def build(res: CallResult, cfg: Config) -> Dict[str, Any]:
    """Собирает заключение по звонку: что сделано, что упущено, что делать."""
    if not res.ok:
        return {"headline": res.skipped_reason or res.error or "не разобран",
                "lines": [], "done": [], "missed": [], "critical": [],
                "advice": [], "text": res.skipped_reason or res.error}

    done = [c for c in res.checks if c.passed is True]
    missed = [c for c in res.checks if c.passed is False]
    unknown = [c for c in res.checks if c.passed is None]
    critical = [c for c in missed if c.required]

    m = res.metrics
    offered = [p.name for p in res.products if p.offered]
    not_offered = [p.name for p in res.products if not p.offered]
    missed_required = [p.name for p in res.products if p.required and not p.offered]

    headline = f"{res.score_pct:.1f}% — {res.grade}"
    if critical:
        headline += f", критичных провалов: {len(critical)}"

    lines: List[str] = [
        f"Выполнено пунктов: {len(done)} из {len(done) + len(missed)}"
        + (f" (ещё {len(unknown)} не проверялись)" if unknown else ""),
    ]
    if critical:
        lines.append("Критично не выполнено: "
                     + ", ".join(f"«{c.title}»" for c in critical))
    if missed and not critical:
        lines.append("Не выполнено: " + ", ".join(f"«{c.title}»" for c in missed[:5]))

    if offered:
        lines.append("Предложены товары: " + ", ".join(offered))
    if missed_required:
        lines.append("НЕ предложены обязательные: " + ", ".join(missed_required))
    elif not_offered:
        lines.append("Не предложены: " + ", ".join(not_offered[:5]))

    op_int = m.get("operator_interruptions", 0)
    if op_int:
        first = next((e for e in m.get("_interruption_events", [])
                      if e["by"] == OPERATOR), None)
        where = f", первое в {_ts(first['at'])}" if first else ""
        lines.append(f"Оператор перебивал клиента: {op_int} раз{where}")

    ratio = m.get("operator_talk_ratio", 0)
    if ratio:
        lines.append(f"Доля речи оператора: {ratio:.0%}"
                     + (" — говорил заметно больше клиента" if ratio > 0.7 else ""))

    if m.get("stop_phrase_hits"):
        ev = m.get("_stop_phrase_evidence", [])
        where = f' («{ev[0].pattern}» в {ev[0].ts()})' if ev else ""
        lines.append(f"Стоп-фразы: {m['stop_phrase_hits']}{where}")

    if m.get("long_pauses"):
        lines.append(f"Затяжных пауз: {m['long_pauses']}, "
                     f"самая длинная {m.get('longest_pause_sec', 0)} с")

    advice = _advice(res, missed, critical, missed_required, m)

    text = headline + ".\n" + "\n".join("• " + s for s in lines)
    if advice:
        text += "\nЧто исправить:\n" + "\n".join("— " + a for a in advice)

    return {"headline": headline, "lines": lines,
            "done": [c.title for c in done], "missed": [c.title for c in missed],
            "critical": [c.title for c in critical], "advice": advice, "text": text}


def _advice(res: CallResult, missed: List[CheckResult], critical: List[CheckResult],
            missed_products: List[str], m: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    by_id = {c.id: c for c in missed}

    if "greeting" in by_id:
        out.append("Начинать разговор с приветствия в первые секунды")
    if "self_intro" in by_id:
        out.append("Представляться по имени в начале разговора")
    if "ask_client_name" in by_id:
        out.append("Спрашивать имя клиента и обращаться по имени")
    if "needs_questions" in by_id:
        out.append("Задавать уточняющие вопросы до презентации товара")
    if "active_listening" in by_id:
        out.append("Проговаривать запрос клиента своими словами: «правильно понимаю…»")
    if "benefits_language" in by_id:
        out.append("Объяснять выгоду для клиента, а не только характеристики")
    if "objection_handling" in by_id:
        out.append("Не оставлять возражение без ответа — приводить аргументы")
    if "next_step" in by_id:
        out.append("Заканчивать договорённостью о следующем шаге")
    if "farewell" in by_id:
        out.append("Прощаться перед завершением разговора")
    if missed_products:
        out.append("Предлагать обязательные позиции: " + ", ".join(missed_products))
    if m.get("operator_interruptions", 0) > 1:
        out.append("Дослушивать клиента до конца, не перебивать")
    if m.get("operator_talk_ratio", 0) > 0.75:
        out.append("Меньше говорить самому, больше слушать клиента")
    if m.get("stop_phrase_hits"):
        out.append("Убрать стоп-фразы из речи")

    for c in critical:
        if c.id not in ("greeting", "self_intro", "next_step", "no_interruptions",
                        "no_stop_phrases", "no_argue", "products_offered"):
            out.append(f"Критичный пункт: {c.title}")
    return out[:8]
