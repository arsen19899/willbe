"""Сквозной разбор звонка: аудио -> текст -> роли -> метрики -> чек-лист -> балл."""
from __future__ import annotations

import os
import re
import time
import traceback
from typing import Callable, List, Optional

from . import asr, audio as au, checks as ck, diarize as dz, llm as llm_mod, metrics as mt
from .config import Config
from .models import CLIENT, OPERATOR, CallResult, Transcript
from .roles import resolve_roles

ProgressCb = Optional[Callable[[str, float], None]]


def _name_filter(path: str, cfg: Config) -> Optional[str]:
    name = os.path.basename(path)

    def match(pat: str) -> bool:
        if pat.lower().startswith("re:"):
            try:
                return re.search(pat[3:], name) is not None
            except re.error:
                return False
        return pat.lower() in name.lower()

    inc = cfg.filters.file_name_include or []
    if inc and not any(match(p) for p in inc):
        return "не подходит под фильтр имени файла"
    for p in (cfg.filters.file_name_exclude or []):
        if match(p):
            return f"исключён фильтром «{p}»"
    return None


def analyze_file(path: str, cfg: Config, progress: ProgressCb = None,
                 transcript_override: Optional[Transcript] = None) -> CallResult:
    """Разбирает один файл. transcript_override — для тестов и повторного
    анализа по готовой стенограмме без повторного распознавания."""
    t0 = time.time()
    res = CallResult(file=os.path.abspath(path) if path else "",
                     file_name=os.path.basename(path) if path else "")

    def say(msg: str, frac: float = 0.0):
        if progress:
            progress(msg, frac)

    try:
        if transcript_override is None:
            skip = _name_filter(path, cfg)
            if skip:
                res.ok, res.skipped_reason = False, skip
                return res

            say(f"{res.file_name}: читаю аудио", 0.02)
            a = au.load_audio(path, dual_mono_corr=cfg.speakers.dual_mono_corr)
            res.duration = a.duration

            if a.duration < cfg.filters.min_duration_sec:
                res.ok = False
                res.skipped_reason = (f"короче {cfg.filters.min_duration_sec:g} с "
                                      f"({a.duration:.1f} с)")
                return res
            if a.duration > cfg.filters.max_duration_sec:
                res.ok = False
                res.skipped_reason = f"длиннее {cfg.filters.max_duration_sec:g} с"
                return res

            sil = au.silence_ratio(a.samples, a.sr)
            if sil > cfg.filters.skip_if_silence_ratio_above:
                res.ok = False
                res.skipped_reason = f"почти полностью тишина ({sil:.0%})"
                return res

            say(f"{res.file_name}: распознаю речь", 0.1)
            tr = asr.transcribe(a, cfg.asr, cfg.speakers, progress)

            if tr.source == "single" and cfg.speakers.mode != "single":
                say(f"{res.file_name}: разделяю голоса", 0.62)
                turns, method, warn = dz.diarize(au.to_mono(a), a.sr, cfg.speakers)
                tr.segments = dz.assign_turns_to_segments(tr.segments, turns)
                tr.source = f"diarization:{method}"
                if warn:
                    res.warnings.append(warn)
        else:
            tr = transcript_override
            res.duration = tr.duration
            sil = 0.0

        say(f"{res.file_name}: определяю роли", 0.68)
        tr, role_warnings = resolve_roles(tr, cfg.speakers)
        res.warnings.extend(role_warnings)
        res.transcript = tr

        total_words = sum(s.word_count for s in tr.segments)
        if total_words < cfg.filters.min_words:
            res.ok = False
            res.skipped_reason = f"распознано мало слов ({total_words})"
            res.transcript = tr
            return res

        say(f"{res.file_name}: считаю метрики", 0.74)
        res.metrics = mt.compute(tr, cfg, silence_ratio=sil)

        say(f"{res.file_name}: проверяю чек-лист", 0.80)
        matcher = ck.Matcher(tr, cfg)
        res.products = ck.evaluate_products(cfg, matcher)
        res.checks = ck.run_rule_checks(cfg, tr, res.metrics, matcher, res.products)
        res.keyword_hits = _keyword_hits(res)

        if cfg.llm.enabled and any(c.type == "llm" for c in cfg.checks):
            say(f"{res.file_name}: смысловая оценка", 0.88)
        summary, recs, extra, warns = llm_mod.evaluate(
            cfg, tr, res.checks, res.products, res.metrics
        )
        res.llm_summary = summary
        res.llm_recommendations = recs
        res.metrics.update({k: v for k, v in (extra or {}).items() if v is not None})
        res.warnings.extend(warns)

        # товары могли быть дозачтены моделью — пересчитываем сводный пункт
        _refresh_product_check(cfg, res)

        res.score, res.score_max, res.score_pct, res.grade, res.category_scores = \
            ck.score(res.checks)
        say(f"{res.file_name}: готово", 1.0)
    except Exception as exc:
        res.ok = False
        res.error = f"{type(exc).__name__}: {exc}"
        res.warnings.append(traceback.format_exc(limit=3))
    finally:
        res.processing_sec = round(time.time() - t0, 1)
    return res


def _refresh_product_check(cfg: Config, res: CallResult) -> None:
    for c in cfg.checks:
        if c.type != "product":
            continue
        target = next((r for r in res.checks if r.id == c.id), None)
        if target is None:
            continue
        required = [p for p in res.products if p.required]
        pool = required or res.products
        missing = [p.name for p in pool if not p.offered]
        offered = [p.name for p in pool if p.offered]
        target.passed = not missing
        target.detail = ("Предложены: " + ", ".join(offered) if offered
                         else "Ничего не предложено")
        if missing:
            target.detail += "; не предложены: " + ", ".join(missing)
        target.evidence = [e for p in res.products if p.offered for e in p.evidence][:5]


def _keyword_hits(res: CallResult) -> List[dict]:
    """Плоский список всех найденных фраз — для листа «Ключевые фразы»."""
    out = []
    for c in res.checks:
        for e in c.evidence:
            out.append({
                "check_id": c.id, "check": c.title, "pattern": e.pattern,
                "how": e.how, "speaker": e.speaker, "at": round(e.start, 2),
                "quote": e.quote,
            })
    for p in res.products:
        for e in p.evidence:
            out.append({
                "check_id": "product", "check": p.name, "pattern": e.pattern,
                "how": e.how, "speaker": e.speaker, "at": round(e.start, 2),
                "quote": e.quote,
            })
    return out


def analyze_many(paths: List[str], cfg: Config, progress: ProgressCb = None,
                 stop_flag: Optional[Callable[[], bool]] = None) -> List[CallResult]:
    out: List[CallResult] = []
    n = max(1, len(paths))
    for i, p in enumerate(paths):
        if stop_flag and stop_flag():
            break

        def sub(msg: str, frac: float, _i=i):
            if progress:
                progress(f"[{_i + 1}/{n}] {msg}", (_i + frac) / n)

        out.append(analyze_file(p, cfg, sub))
    return out
