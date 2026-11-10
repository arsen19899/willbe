"""Локальный веб-интерфейс CallQA (FastAPI + одна HTML-страница)."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
import webbrowser
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .. import audio as au, report, verdict as vd
from ..config import DEFAULT_CONFIG_PATH, Config
from ..metrics import METRIC_TITLES
from ..models import ROLE_RU, CallResult
from ..pipeline import analyze_many

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STATE: Dict[str, Any] = {"config_path": DEFAULT_CONFIG_PATH}
JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

app = FastAPI(title="CallQA", docs_url=None, redoc_url=None)


def _cfg() -> Config:
    return Config.load(STATE["config_path"])


# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/config")
def get_config():
    cfg = _cfg()
    return {
        "path": STATE["config_path"],
        "yaml": yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False,
                               width=100),
        "config": cfg.to_dict(),
        "metric_titles": METRIC_TITLES,
    }


@app.post("/api/config")
def set_config(payload: Dict[str, Any] = Body(...)):
    """Принимает либо целиком YAML, либо структуру из редактора настроек."""
    try:
        if "yaml" in payload:
            data = yaml.safe_load(payload["yaml"]) or {}
        else:
            data = payload.get("config") or {}
        cfg = Config.from_dict(data)
    except Exception as exc:
        raise HTTPException(400, f"Некорректные настройки: {exc}")
    save_to = payload.get("path") or STATE["config_path"]
    if os.path.abspath(save_to) == os.path.abspath(DEFAULT_CONFIG_PATH):
        # не затираем эталонный конфиг — пишем рядом
        save_to = os.path.join(os.path.dirname(DEFAULT_CONFIG_PATH), "current.yaml")
    cfg.save(save_to)
    STATE["config_path"] = save_to
    return {"ok": True, "path": save_to, "warnings": cfg.warnings}


@app.post("/api/config/reset")
def reset_config():
    STATE["config_path"] = DEFAULT_CONFIG_PATH
    return get_config()


@app.get("/api/env")
def env_check():
    from ..text import fuzzy_available, morphology_available

    def has(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:
            return False

    from ..cli import _gpu_hardware

    # Видеокарту спрашиваем у драйвера, а не у torch: для распознавания
    # torch не нужен вовсе, и по нему было бы видно «видеокарты нет»,
    # даже когда CUDA прекрасно работает.
    gpu_name = _gpu_hardware()
    cuda_asr = False
    try:
        import ctranslate2

        cuda_asr = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass

    cfg = _cfg()
    return {
        "faster_whisper": has("faster_whisper"),
        "gpu": bool(gpu_name),
        "gpu_name": gpu_name,
        "cuda_asr": cuda_asr,
        "ffmpeg": bool(au.ffmpeg_path()),
        "pymorphy": morphology_available(),
        "rapidfuzz": fuzzy_available(),
        "pyannote": has("pyannote.audio"),
        "torch": has("torch"),
        "llm": bool(cfg.llm.enabled and cfg.llm_api_key()),
        "llm_local": cfg.llm.is_local(),
        "llm_model": cfg.llm.model,
        "device": cfg.asr.device,
        "compute_type": cfg.asr.compute_type,
        "asr_model": cfg.asr.model,
    }


@app.get("/api/browse")
def browse(path: str = ""):
    """Простой обзор папок на этой машине — чтобы брать записи по пути, без загрузки."""
    path = path or os.path.expanduser("~")
    if not os.path.isdir(path):
        raise HTTPException(400, "Не папка")
    dirs, files = [], []
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                dirs.append({"name": name, "path": full})
            elif au.is_audio_file(full):
                files.append({"name": name, "path": full,
                              "size": os.path.getsize(full)})
    except PermissionError:
        raise HTTPException(403, "Нет доступа к папке")
    return {"path": os.path.abspath(path),
            "parent": os.path.dirname(os.path.abspath(path)),
            "dirs": dirs, "files": files}


# --------------------------------------------------------------------------- #
@app.post("/api/analyze/upload")
async def analyze_upload(files: List[UploadFile] = File(...)):
    tmp = tempfile.mkdtemp(prefix="callqa_up_")
    paths = []
    for f in files:
        dest = os.path.join(tmp, os.path.basename(f.filename or "call.wav"))
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(dest)
    return _start_job(paths, cleanup=tmp)


@app.post("/api/analyze/path")
def analyze_path(payload: Dict[str, Any] = Body(...)):
    src = payload.get("path") or ""
    recursive = bool(payload.get("recursive", True))
    if not os.path.exists(src):
        raise HTTPException(400, f"Путь не найден: {src}")
    paths = au.list_audio_files(src, recursive=recursive)
    if not paths:
        raise HTTPException(400, "Аудиофайлы не найдены по этому пути")
    return _start_job(paths)


def _start_job(paths: List[str], cleanup: Optional[str] = None):
    job_id = uuid.uuid4().hex[:12]
    cfg = _cfg()
    job = {"id": job_id, "status": "running", "progress": 0.0,
           "message": "запуск", "total": len(paths), "done": 0,
           "results": None, "error": "", "stop": False,
           "started": datetime.now().isoformat(timespec="seconds"),
           "files": [os.path.basename(p) for p in paths]}
    with _LOCK:
        JOBS[job_id] = job

    def progress(msg: str, frac: float):
        job["message"] = msg
        job["progress"] = round(min(1.0, max(0.0, frac)), 3)

    def worker():
        try:
            results = analyze_many(paths, cfg, progress=progress,
                                   stop_flag=lambda: job["stop"])
            job["results"] = results
            written = report.write_all(results, cfg, cfg.output.dir,
                                       f"callqa_{job_id}")
            job["reports"] = written
            job["table"] = _table(results, cfg)
            job["status"] = "done"
            job["progress"] = 1.0
            job["message"] = "готово"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if cleanup and os.path.isdir(cleanup):
                shutil.rmtree(cleanup, ignore_errors=True)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "total": len(paths)}


@app.post("/api/demo")
def analyze_demo():
    """Разбор встроенного примера — чтобы увидеть формат отчёта без записей и моделей."""
    from .. import demo

    cfg = _cfg()
    job_id = "demo" + uuid.uuid4().hex[:8]
    results = demo.run(cfg)
    written = report.write_all(results, cfg, cfg.output.dir, f"callqa_{job_id}")
    with _LOCK:
        JOBS[job_id] = {"id": job_id, "status": "done", "progress": 1.0,
                        "message": "готово", "total": 1, "done": 1,
                        "results": results, "reports": written,
                        "table": _table(results, cfg), "error": "", "stop": False,
                        "started": datetime.now().isoformat(timespec="seconds"),
                        "files": ["demo"]}
    return {"job_id": job_id, "total": 1}


# --------------------------------------------------------------------------- #
#  Обучение правил на размеченных записях
# --------------------------------------------------------------------------- #
@app.post("/api/learn")
def start_learn(payload: Dict[str, Any] = Body(...)):
    from .. import learn as lr

    good = (payload.get("good") or "").strip()
    bad = (payload.get("bad") or "").strip()
    for folder, name in ((good, "успешных"), (bad, "неуспешных")):
        if not folder or not os.path.isdir(folder):
            raise HTTPException(400, f"Не найдена папка {name} разговоров: {folder}")

    cfg = _cfg()
    cfg.llm.enabled = False          # для обучения смысловая оценка не нужна
    job_id = "learn" + uuid.uuid4().hex[:8]
    job = {"id": job_id, "status": "running", "progress": 0.0,
           "message": "запуск", "error": "", "stop": False, "result": None,
           "started": datetime.now().isoformat(timespec="seconds")}
    with _LOCK:
        JOBS[job_id] = job

    def progress(msg: str, frac: float):
        job["message"] = msg
        job["progress"] = round(min(1.0, max(0.0, frac)), 3)

    def worker():
        try:
            res = lr.analyse(
                good, bad, cfg, progress=progress,
                min_calls=int(payload.get("min_calls") or 3),
                max_q=float(payload.get("max_q") or 0.10),
                min_lift=float(payload.get("min_lift") or 1.6),
                stop_flag=lambda: job["stop"],
            )
            job["learn_result"] = res
            job["result"] = res.to_dict()
            job["status"] = "done"
            job["progress"] = 1.0
            job["message"] = "готово"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/learn/{job_id}")
def learn_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    return {k: v for k, v in job.items()
            if k not in ("learn_result", "results", "table")}


@app.post("/api/learn/{job_id}/apply")
def learn_apply(job_id: str, payload: Dict[str, Any] = Body(...)):
    """Добавляет выбранные находки в чек-лист. Существующие пункты не трогает."""
    from .. import learn as lr

    job = JOBS.get(job_id)
    if not job or not job.get("learn_result"):
        raise HTTPException(404, "Результат обучения не найден")

    checks = lr.build_checks(
        job["learn_result"],
        phrase_ids=payload.get("phrases"),
        metric_ids=payload.get("metrics"),
    )
    if not checks:
        raise HTTPException(400, "Не выбрано ни одного пункта")

    cfg = _cfg()
    before = len(cfg.checks)
    merged = lr.merge_into_config(cfg, checks)
    target = os.path.join(os.path.dirname(DEFAULT_CONFIG_PATH), "current.yaml")
    merged.save(target)
    STATE["config_path"] = target
    return {"ok": True, "added": len(merged.checks) - before, "path": target,
            "titles": [c["title"] for c in checks]}


@app.post("/api/learn/{job_id}/preview")
def learn_preview(job_id: str, payload: Dict[str, Any] = Body(...)):
    """YAML выбранных пунктов — чтобы посмотреть и поправить руками."""
    from .. import learn as lr

    job = JOBS.get(job_id)
    if not job or not job.get("learn_result"):
        raise HTTPException(404, "Результат обучения не найден")
    checks = lr.build_checks(job["learn_result"],
                             phrase_ids=payload.get("phrases"),
                             metric_ids=payload.get("metrics"))
    return {"yaml": yaml.safe_dump({"checks": checks}, allow_unicode=True,
                                   sort_keys=False, width=100),
            "count": len(checks)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    return {k: v for k, v in job.items() if k not in ("results",)}


@app.post("/api/jobs/{job_id}/stop")
def job_stop(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    job["stop"] = True
    return {"ok": True}


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    if job["status"] != "done":
        raise HTTPException(409, "Задача ещё не завершена")
    return JSONResponse(job["table"])


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str, fmt: str = "xlsx"):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Отчёт недоступен")
    path = (job.get("reports") or {}).get(fmt, "")
    if not path or not os.path.exists(path):
        raise HTTPException(404, f"Файл отчёта ({fmt}) не найден")
    return FileResponse(path, filename=os.path.basename(path))


@app.get("/api/jobs/{job_id}/transcript")
def job_transcript(job_id: str, file: str):
    job = JOBS.get(job_id)
    if not job or not job.get("results"):
        raise HTTPException(404, "Нет данных")
    for r in job["results"]:
        if r.file_name == file and r.transcript:
            return {"file": file,
                    "segments": [{"start": s.start, "end": s.end,
                                  "speaker": ROLE_RU.get(s.speaker, s.speaker),
                                  "text": s.text} for s in
                                 sorted(r.transcript.segments, key=lambda x: x.start)]}
    raise HTTPException(404, "Транскрипт не найден")


# --------------------------------------------------------------------------- #
def _table(results: List[CallResult], cfg: Config) -> Dict[str, Any]:
    header, rows, statuses = report.summary_rows(results, cfg)
    return {
        "header": header,
        "rows": rows,
        "statuses": statuses,
        "calls": [{
            "file": r.file_name,
            "ok": r.ok,
            "reason": r.skipped_reason or r.error,
            "duration": round(r.duration, 1),
            "score_pct": r.score_pct,
            "grade": r.grade,
            "summary": r.llm_summary,
            "recommendations": r.llm_recommendations,
            "warnings": [w for w in r.warnings if "Traceback" not in w],
            "categories": r.category_scores,
            "metrics": {METRIC_TITLES.get(k, k): v for k, v in r.metrics.items()
                        if not k.startswith("_")},
            "verdict": vd.build(r, cfg),
            "findings": vd.findings(r),
            "checks": [{
                "id": c.id, "title": c.title, "category": c.category,
                "type": c.type, "speaker": ROLE_RU.get(c.speaker, c.speaker),
                "weight": c.weight, "required": c.required, "status": c.status_ru,
                "detail": c.detail, "cell": vd.cell(c), "state": vd.status_of(c),
                "evidence": [{"quote": e.quote, "at": e.ts(), "sec": round(e.start, 1),
                              "how": vd._how_ru(e.how), "pattern": e.pattern,
                              "matched": e.matched} for e in c.evidence],
            } for c in r.checks],
            "products": [{"name": p.name, "group": p.group, "required": p.required,
                          "offered": p.offered, "how": vd._how_ru(p.how),
                          "cell": vd.product_cell(p),
                          "evidence": [{"quote": e.quote, "at": e.ts(),
                                        "sec": round(e.start, 1),
                                        "matched": e.matched, "pattern": e.pattern}
                                       for e in p.evidence]} for p in r.products],
            "interruptions": [{"at": e["at"], "by": ROLE_RU.get(e["by"], e["by"]),
                               "of": ROLE_RU.get(e["of"], e["of"]),
                               "overlap": e["overlap_sec"], "quote": e["quote"],
                               "interrupted": e["interrupted_quote"]}
                              for e in r.metrics.get("_interruption_events", [])],
        } for r in results],
    }


def run(host: str = "127.0.0.1", port: int = 8765,
        config_path: str = DEFAULT_CONFIG_PATH, open_browser: bool = True) -> None:
    import uvicorn

    STATE["config_path"] = config_path
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"CallQA открыт: {url}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
