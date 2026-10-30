import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core import pipeline
from core.manifest import ManifestError, load_manifest
from report import build_html

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
STATIC = ROOT / "server" / "static"
RUN_ID_RE = pipeline.re.compile(r"^[A-Za-z0-9_\-]+$") if hasattr(pipeline, "re") else __import__("re").compile(r"^[A-Za-z0-9_\-]+$")
MAX_LOG_CHUNK = 64 * 1024
LOG_NAME = "run.log"

app = FastAPI(title="DubQA v2", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(400, "bad run_id")
    d = RUNS / run_id
    if not d.exists():
        raise HTTPException(404, "run not found")
    return d


class _ThreadFilter(logging.Filter if "logging" in globals() else __import__("logging").Filter):
    def __init__(self, thread_name: str):
        super().__init__()
        self.thread_name = thread_name

    def filter(self, record):
        return record.threadName == self.thread_name


def _job(run_dir: Path, mode: str, use_judge: bool, use_gemini: bool) -> None:
    import logging
    thread_name = threading.current_thread().name
    handler = logging.FileHandler(run_dir / LOG_NAME, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler.addFilter(_ThreadFilter(thread_name))
    logger = logging.getLogger("dubqa")
    logger.addHandler(handler)
    try:
        pipeline.run(run_dir, mode=mode, use_judge=use_judge, use_gemini=use_gemini)
        build_html.build(run_dir)
        logging.getLogger("dubqa").info("html report built")
        try:
            from core.qa_report_bundle import export_run_reports
            export_run_reports(run_dir)
            logging.getLogger("dubqa").info("excel and word reports built")
            _update_meta(run_dir, report_status="success")
        except Exception as rep_err:
            logging.getLogger("dubqa").warning("excel/word report generation error: %s", rep_err)
            _update_meta(run_dir, report_status="failed", report_error=str(rep_err))
    except Exception:
        pass
    finally:
        logger.removeHandler(handler)
        handler.close()


def _update_meta(run_dir: Path, **kwargs) -> None:
    p = run_dir / "run_meta.json"
    meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    meta.update(kwargs)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html", media_type="text/html")


@app.get("/api/platforms")
def platforms():
    data = yaml.safe_load((ROOT / "config" / "vendors.yaml").read_text(encoding="utf-8")) or {}
    items = data.get("vendors", {})
    return [name for name, _ in sorted(items.items(), key=lambda kv: kv[1].get("order", 99))]


@app.post("/api/runs", status_code=202)
async def create_run(manifest: UploadFile = File(...), files: list[UploadFile] = File(...),
                     mode: str = Form("precomputed"), use_judge: bool = Form(False),
                     use_gemini: bool = Form(True)):
    if mode not in ("precomputed", "full"):
        raise HTTPException(400, "mode must be precomputed or full")
    raw = await manifest.read()
    try:
        data = json.loads(raw)
        load_manifest_from_dict(data)
    except (ValueError, ManifestError) as e:
        raise HTTPException(400, f"manifest invalid: {e}")
    run_id = data["run_id"]
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(400, "bad run_id in manifest")
    run_dir = RUNS / run_id
    if run_dir.exists():
        raise HTTPException(409, "run_id already exists")
    for sub in ("inputs", "work", "measure", "judge", "precomputed", "report", "lips", "words"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_bytes(raw)
    wanted = {data["source"]["file"], *(o["file"] for o in data["outputs"])}
    saved = []
    for f in files:
        name = Path(f.filename or "").name
        if not name or name != (f.filename or ""):
            raise HTTPException(400, f"bad file name: {f.filename}")
        if name.endswith(".json"):
            dest = run_dir / "precomputed" / name
        elif name in wanted:
            dest = run_dir / "inputs" / name
        else:
            raise HTTPException(400, f"unexpected file: {name}")
        dest.write_bytes(await f.read())
        saved.append(name)
    missing = [w for w in wanted if w not in saved]
    if missing:
        raise HTTPException(400, f"missing media files: {missing}")
    _write_meta(run_dir, "queued", mode=mode, created=datetime.now(timezone.utc).isoformat())
    (run_dir / LOG_NAME).write_text(
        f"{datetime.now(timezone.utc).isoformat()} INFO run {run_id} queued (mode={mode})\n", encoding="utf-8")
    t = threading.Thread(target=_job, args=(run_dir, mode, use_judge, use_gemini), name=f"run-{run_id}", daemon=True)
    t.start()
    return {"run_id": run_id, "status": "queued", "status_url": f"/api/runs/{run_id}"}


def load_manifest_from_dict(data: dict) -> None:
    from core.manifest import validate_manifest
    validate_manifest(data)


def _write_meta(run_dir: Path, status: str, **extra) -> None:
    p = run_dir / "run_meta.json"
    meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    meta.update({"status": status}, **extra)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


@app.get("/api/runs/{run_id}")
def status(run_id: str):
    d = _run_dir(run_id)
    p = d / "run_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "queued"}


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str, offset: int = Query(0, ge=0)):
    p = _run_dir(run_id) / LOG_NAME
    if not p.exists():
        return {"text": "", "offset": 0}
    size = p.stat().st_size
    start = min(offset, size)
    with p.open("rb") as f:
        f.seek(start)
        chunk = f.read(MAX_LOG_CHUNK)
    return {"text": chunk.decode("utf-8", errors="replace"), "offset": start + len(chunk)}


@app.get("/api/runs/{run_id}/results")
def results(run_id: str):
    p = _run_dir(run_id) / "results.json"
    if not p.exists():
        raise HTTPException(404, "results not ready")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/runs/{run_id}/report")
def report(run_id: str):
    p = _run_dir(run_id) / "report" / "index.html"
    if not p.exists():
        raise HTTPException(404, "report not ready")
    return FileResponse(p, media_type="text/html")


@app.get("/api/runs/{run_id}/report/excel")
def report_excel(run_id: str):
    p = _run_dir(run_id) / "report" / "QA_Results.xlsx"
    if not p.exists():
        raise HTTPException(404, "Excel workbook not ready")
    return FileResponse(p, filename=f"{run_id}_QA_Results.xlsx",
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/api/runs/{run_id}/report/word")
def report_word(run_id: str):
    p = _run_dir(run_id) / "report" / "QA_Report.docx"
    if not p.exists():
        raise HTTPException(404, "Word report not ready")
    return FileResponse(p, filename=f"{run_id}_QA_Report.docx",
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@app.get("/api/runs/{run_id}/report/{asset:path}")
def report_asset(run_id: str, asset: str):
    base = (_run_dir(run_id) / "report").resolve()
    p = (base / asset).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(p)
