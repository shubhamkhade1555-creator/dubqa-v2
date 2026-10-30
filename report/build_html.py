"""results.json -> report/index.html (self-contained except for media files)."""
import json
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from core import media, scoring

HERE = Path(__file__).resolve().parent


def build(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    rep = run_dir / "report"
    media_dir = rep / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    src_name = res["source"]["file"]
    src_file = run_dir / "inputs" / src_name
    if src_file.exists():
        media.make_web_copy(src_file, media_dir / f"source__{src_name.rsplit('.', 1)[0]}.mp4")
    for oid, o in res["outputs"].items():
        f = run_dir / "inputs" / o["file"]
        if f.exists():
            media.make_web_copy(f, media_dir / f"{oid}.mp4")
    env = Environment(loader=FileSystemLoader(HERE / "templates"),
                      autoescape=select_autoescape(["html"]))
    tpl = env.get_template("report.html")
    lips = {}
    for oid in res["outputs"]:
        p = run_dir / "precomputed" / f"{oid}_lips.json"
        if p.exists():
            lips[oid] = json.loads(p.read_text(encoding="utf-8")).get("opening", [])
    th = scoring.load_thresholds()
    html = tpl.render(r=res, lips_json=json.dumps(lips), metrics=th["metrics"],
                      src_media=f"media/source__{src_name.rsplit('.', 1)[0]}.mp4",
                      outputs=list(res["outputs"].values()))
    out = rep / "index.html"
    out.write_text(html, encoding="utf-8")
    return out
