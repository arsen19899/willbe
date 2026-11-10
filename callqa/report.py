"""Формирование отчётов: Excel (несколько листов), CSV и JSON."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from typing import Optional

from . import verdict
from .config import Config
from .metrics import METRIC_TITLES
from .models import CLIENT, OPERATOR, ROLE_RU, CallResult, _ts

STATUS_YES = "было"
STATUS_NO = "не было"
STATUS_NA = "н/д"

_GREEN = "FFC6EFCE"
_GREEN_TXT = "FF006100"
_RED = "FFFFC7CE"
_RED_TXT = "FF9C0006"
_GREY = "FFF2F2F2"
_HEAD = "FF305496"


# --------------------------------------------------------------------------- #
#  Табличное представление (общее для xlsx/csv/веб-интерфейса)
# --------------------------------------------------------------------------- #
def summary_rows(results: List[CallResult], cfg: Config
                 ) -> (List[str], List[List[Any]], List[List[Optional[str]]]):
    """Сводная таблица.

    В колонке каждого пункта не «было/не было», а конкретная сработавшая
    фраза с таймкодом: «меня зовут» 0:07. Третьим значением возвращается
    матрица статусов (yes/no/na/None) — по ней Excel красит ячейки.
    """
    check_titles = [c.title for c in cfg.checks]
    check_ids = [c.id for c in cfg.checks]
    product_names = [p.name for p in cfg.products]

    header = (["Файл", "Длит., мин", "Статус", "Балл, %", "Оценка", "Заключение",
               "Критичных провалов", "Перебиваний оператором", "Перебиваний клиентом",
               "Доля речи оператора", "Стоп-фразы"]
              + check_titles
              + [f"Товар: {n}" for n in product_names]
              + ["Что исправить", "Резюме модели", "Замечания"])
    n_fixed = 11

    rows: List[List[Any]] = []
    stats: List[List[Optional[str]]] = []
    for r in results:
        if not r.ok:
            status = (f"Пропущен: {r.skipped_reason}" if r.skipped_reason
                      else f"Ошибка: {r.error}")
            rows.append([r.file_name, round(r.duration / 60, 2) if r.duration else "",
                         status, "", "", "", "", "", "", "", ""]
                        + [STATUS_NA] * len(check_titles)
                        + [STATUS_NA] * len(product_names)
                        + ["", "", "; ".join(r.warnings[:2])])
            stats.append([None] * n_fixed
                         + ["na"] * (len(check_titles) + len(product_names))
                         + [None] * 3)
            continue

        by_id = {c.id: c for c in r.checks}
        prod_by_name = {p.name: p for p in r.products}
        v = verdict.build(r, cfg)
        crit = len(v["critical"])
        m = r.metrics

        rows.append(
            [r.file_name,
             round(r.duration / 60, 2),
             "Обработан",
             r.score_pct,
             r.grade,
             v["headline"],
             crit,
             m.get("operator_interruptions", ""),
             m.get("client_interruptions", ""),
             m.get("operator_talk_ratio", ""),
             m.get("stop_phrase_hits", "")]
            + [verdict.cell(by_id[i]) if i in by_id else STATUS_NA for i in check_ids]
            + [verdict.product_cell(prod_by_name[n]) if n in prod_by_name else STATUS_NA
               for n in product_names]
            + [" • ".join(v["advice"]),
               r.llm_summary,
               "; ".join(w for w in r.warnings if "Traceback" not in w)[:500]]
        )
        stats.append(
            [None] * n_fixed
            + [verdict.status_of(by_id[i]) if i in by_id else "na" for i in check_ids]
            + [("yes" if prod_by_name[n].offered else "no") if n in prod_by_name else "na"
               for n in product_names]
            + [None] * 3
        )
    return header, rows, stats


def verdict_rows(results: List[CallResult], cfg: Config) -> (List[str], List[List[Any]]):
    header = ["Файл", "Балл, %", "Оценка", "Заключение", "Выполнено",
              "Не выполнено", "Критичные провалы", "Что исправить"]
    rows: List[List[Any]] = []
    for r in results:
        v = verdict.build(r, cfg)
        rows.append([
            r.file_name,
            r.score_pct if r.ok else "",
            r.grade if r.ok else "",
            "\n".join(v["lines"]) if v["lines"] else v["headline"],
            ", ".join(v["done"]),
            ", ".join(v["missed"]),
            ", ".join(v["critical"]),
            "\n".join(v["advice"]),
        ])
    return header, rows


def findings_rows(results: List[CallResult]) -> (List[str], List[List[Any]]):
    """Журнал находок: каждое срабатывание с секундой и цитатой, по порядку."""
    header = ["Файл", "Время", "Секунда", "Тип", "Что сработало",
              "Найдено в записи", "Что искали", "Кто говорит", "Как найдено",
              "Цитата", "Результат"]
    rows: List[List[Any]] = []
    for r in results:
        for f in verdict.findings(r):
            rows.append([r.file_name, f["time"], f["sec"], f["kind"], f["what"],
                         f["pattern"], f.get("searched", ""), f["speaker"],
                         f["how"], f["quote"], f["result"]])
    return header, rows


def _prod(r: CallResult, name: str) -> bool:
    return any(p.name == name and p.offered for p in r.products)


def detail_rows(results: List[CallResult]) -> (List[str], List[List[Any]]):
    """По строке на каждое найденное совпадение, а не только на первое."""
    header = ["Файл", "Категория", "Пункт", "Тип", "Чья речь", "Вес", "Критичный",
              "Результат", "Обоснование", "Найденная фраза", "Таймкод", "Секунда",
              "Цитата", "Как найдено"]
    rows: List[List[Any]] = []
    for r in results:
        for c in r.checks:
            base = [r.file_name, c.category, c.title, c.type,
                    ROLE_RU.get(c.speaker, c.speaker), c.weight,
                    "да" if c.required else "", c.status_ru, c.detail]
            if not c.evidence:
                rows.append(base + ["", "", "", "", ""])
                continue
            for ev in c.evidence:
                rows.append(base + [ev.pattern, ev.ts(), round(ev.start, 1),
                                    ev.quote, verdict._how_ru(ev.how)])
    return header, rows


def product_rows(results: List[CallResult]) -> (List[str], List[List[Any]]):
    header = ["Файл", "Товар/услуга", "Группа", "Обязателен", "Предложен",
              "Найденная фраза", "Таймкод", "Секунда", "Цитата", "Как найдено"]
    rows: List[List[Any]] = []
    for r in results:
        for p in r.products:
            ev = p.evidence[0] if p.evidence else None
            rows.append([r.file_name, p.name, p.group, "да" if p.required else "",
                         STATUS_YES if p.offered else STATUS_NO,
                         ev.pattern if ev else "", ev.ts() if ev else "",
                         round(ev.start, 1) if ev else "",
                         ev.quote if ev else "", verdict._how_ru(p.how)])
    return header, rows


def metric_rows(results: List[CallResult]) -> (List[str], List[List[Any]]):
    keys = [k for k in METRIC_TITLES if any(k in r.metrics for r in results)]
    header = ["Файл"] + [METRIC_TITLES[k] for k in keys]
    rows = [[r.file_name] + [r.metrics.get(k, "") for k in keys]
            for r in results if r.ok]
    return header, rows


def interruption_rows(results: List[CallResult]) -> (List[str], List[List[Any]]):
    header = ["Файл", "Таймкод", "Кто перебил", "Кого", "Наложение, сек",
              "Реплика перебившего", "Что перебили"]
    rows: List[List[Any]] = []
    for r in results:
        for e in r.metrics.get("_interruption_events", []):
            rows.append([r.file_name, _ts(e["at"]), ROLE_RU.get(e["by"], e["by"]),
                         ROLE_RU.get(e["of"], e["of"]), e["overlap_sec"],
                         e["quote"], e["interrupted_quote"]])
    return header, rows


def transcript_rows(results: List[CallResult]) -> (List[str], List[List[Any]]):
    header = ["Файл", "Начало", "Конец", "Кто", "Реплика"]
    rows: List[List[Any]] = []
    for r in results:
        if not r.transcript:
            continue
        for s in sorted(r.transcript.segments, key=lambda x: x.start):
            rows.append([r.file_name, _ts(s.start), _ts(s.end),
                         ROLE_RU.get(s.speaker, s.speaker), s.text.strip()])
    return header, rows


# --------------------------------------------------------------------------- #
#  Excel
# --------------------------------------------------------------------------- #
def write_xlsx(results: List[CallResult], cfg: Config, path: str) -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    head_fill = PatternFill("solid", fgColor=_HEAD)
    head_font = Font(color="FFFFFFFF", bold=True, size=10)
    yes_fill = PatternFill("solid", fgColor=_GREEN)
    no_fill = PatternFill("solid", fgColor=_RED)
    na_fill = PatternFill("solid", fgColor=_GREY)

    def sheet(ws, header, rows, freeze="B2", widths=None, wrap_cols=(),
              statuses=None):
        ws.append(header)
        for c in range(1, len(header) + 1):
            cell = ws.cell(row=1, column=c)
            cell.fill = head_fill
            cell.font = head_font
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 42
        for row in rows:
            ws.append(["" if v is None else v for v in row])
        for c in range(1, len(header) + 1):
            letter = get_column_letter(c)
            if widths and c <= len(widths) and widths[c - 1]:
                ws.column_dimensions[letter].width = widths[c - 1]
            else:
                longest = max([len(str(header[c - 1]))] +
                              [len(str(r[c - 1])) for r in rows[:200]
                               if c - 1 < len(r)] or [10])
                ws.column_dimensions[letter].width = min(46, max(10, longest + 2))
        for c in wrap_cols:
            for row in range(2, ws.max_row + 1):
                ws.cell(row=row, column=c).alignment = Alignment(wrap_text=True,
                                                                 vertical="top")
        ws.freeze_panes = freeze
        ws.auto_filter.ref = ws.dimensions

        fills = {"yes": (yes_fill, _GREEN_TXT), "no": (no_fill, _RED_TXT),
                 "na": (na_fill, None)}
        if statuses:
            # цвет берём из матрицы статусов: в ячейке теперь фраза с таймкодом,
            # а не слово «было», по которому раньше определялся цвет
            for ri, row_status in enumerate(statuses, start=2):
                for ci, st in enumerate(row_status, start=1):
                    if st not in fills:
                        continue
                    fill, colour = fills[st]
                    cell = ws.cell(row=ri, column=ci)
                    cell.fill = fill
                    if colour:
                        cell.font = Font(color=colour, size=10)
        else:
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if cell.value == STATUS_YES:
                        cell.fill = yes_fill
                        cell.font = Font(color=_GREEN_TXT, size=10)
                    elif cell.value == STATUS_NO:
                        cell.fill = no_fill
                        cell.font = Font(color=_RED_TXT, size=10)
                    elif cell.value == STATUS_NA:
                        cell.fill = na_fill
        return ws

    ws = wb.active
    ws.title = "Итог"
    h, r = verdict_rows(results, cfg)
    sheet(ws, h, r, freeze="B2", widths=[32, 9, 14, 62, 44, 40, 30, 52],
          wrap_cols=(4, 5, 6, 7, 8))

    h, r, st = summary_rows(results, cfg)
    sheet(wb.create_sheet("Свод"), h, r, freeze="C2",
          widths=[32, 10, 16, 8, 13, 30, 10, 12, 12, 12, 10],
          wrap_cols=(6, len(h) - 2, len(h) - 1, len(h)), statuses=st)

    h, r = findings_rows(results)
    sheet(wb.create_sheet("Находки"), h, r, freeze="A2",
          widths=[28, 9, 9, 18, 30, 28, 26, 12, 24, 50, 14], wrap_cols=(10,))

    h, r = detail_rows(results)
    sheet(wb.create_sheet("Детали"), h, r, freeze="A2",
          widths=[28, 18, 28, 10, 12, 6, 10, 12, 44, 26, 9, 9, 46, 20],
          wrap_cols=(9, 13))

    h, r = product_rows(results)
    sheet(wb.create_sheet("Товары"), h, r,
          widths=[28, 24, 14, 12, 12, 26, 9, 9, 46, 20], wrap_cols=(9,))

    h, r = metric_rows(results)
    sheet(wb.create_sheet("Метрики"), h, r)

    h, r = interruption_rows(results)
    sheet(wb.create_sheet("Перебивания"), h, r, widths=[30, 10, 14, 14, 12, 50, 50],
          wrap_cols=(6, 7))

    if cfg.output.transcript_in_report:
        h, r = transcript_rows(results)
        sheet(wb.create_sheet("Транскрипт"), h, r, widths=[30, 10, 10, 12, 100],
              wrap_cols=(5,))

    _legend(wb, cfg, results)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    wb.save(path)
    return path


def _legend(wb, cfg: Config, results: List[CallResult]) -> None:
    from openpyxl.styles import Font

    ws = wb.create_sheet("О разборе")
    ok = [r for r in results if r.ok]
    lines = [
        ("CallQA — отчёт по качеству звонков", True),
        (f"Сформирован: {datetime.now():%d.%m.%Y %H:%M}", False),
        ("", False),
        (f"Файлов обработано: {len(ok)} из {len(results)}", False),
        (f"Средний балл: {round(sum(r.score_pct for r in ok) / len(ok), 1) if ok else 0} %",
         False),
        (f"Модель распознавания: {cfg.asr.model}", False),
        (f"Смысловая оценка: {cfg.llm.model if cfg.llm.enabled else 'выключена'}", False),
        ("", False),
        ("Как читать отчёт", True),
        ("Свод — по строке на звонок: балл, метрики и колонка на каждый пункт чек-листа.", False),
        ("Детали — по строке на пункт: обоснование, таймкод и цитата.", False),
        ("Товары — что из заданного списка оператор предложил.", False),
        ("Метрики — числовые показатели диалога.", False),
        ("Перебивания — каждый случай наложения речи с цитатами.", False),
        ("Транскрипт — полная расшифровка с ролями.", False),
        ("", False),
        ("«было» — пункт выполнен · «не было» — не выполнен · «н/д» — проверить не удалось",
         False),
        ("", False),
        ("Как найдено (колонка в Деталях):", True),
        ("exact — дословное совпадение · lemma — совпадение по форме слова", False),
        ("fuzzy — с поправкой на ошибки распознавания · regex — по регулярному выражению",
         False),
        ("metric — по числовому порогу · llm — смысловая оценка моделью", False),
    ]
    for text, bold in lines:
        ws.append([text])
        if bold:
            ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=12)
    ws.column_dimensions["A"].width = 110


# --------------------------------------------------------------------------- #
#  CSV / JSON
# --------------------------------------------------------------------------- #
def write_csv(results: List[CallResult], cfg: Config, path: str) -> str:
    import csv

    header, rows, _st = summary_rows(results, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)
    return path


def write_json(results: List[CallResult], cfg: Config, path: str) -> str:
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"asr_model": cfg.asr.model, "llm_model": cfg.llm.model,
                   "checks": [c.id for c in cfg.checks],
                   "products": [p.name for p in cfg.products]},
        "calls": [_clean(r.to_dict(with_transcript=cfg.output.save_transcript))
                  for r in results],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
    m = d.get("metrics") or {}
    d["metrics"] = {k: v for k, v in m.items()
                    if not k.startswith("_") or k == "_interruption_events"}
    return d


def write_all(results: List[CallResult], cfg: Config, out_dir: Optional[str] = None,
              stem: Optional[str] = None) -> Dict[str, str]:
    out_dir = out_dir or cfg.output.dir
    stem = stem or f"callqa_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}
    for fmt in cfg.output.formats:
        p = os.path.join(out_dir, f"{stem}.{fmt}")
        try:
            if fmt == "xlsx":
                written["xlsx"] = write_xlsx(results, cfg, p)
            elif fmt == "csv":
                written["csv"] = write_csv(results, cfg, p)
            elif fmt == "json":
                written["json"] = write_json(results, cfg, p)
        except Exception as exc:  # один формат не должен ронять остальные
            written[fmt] = f"ОШИБКА: {exc}"
    if cfg.output.save_transcript:
        tdir = os.path.join(out_dir, "transcripts")
        os.makedirs(tdir, exist_ok=True)
        for r in results:
            if not r.transcript:
                continue
            base = os.path.splitext(r.file_name)[0] or "call"
            with open(os.path.join(tdir, base + ".txt"), "w", encoding="utf-8") as f:
                f.write(r.transcript.dialogue())
    return written
