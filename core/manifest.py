"""Read and validate a batch manifest."""
import json
import re
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9_\-]+$")


class ManifestError(ValueError):
    pass


def load_manifest(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(data)
    return data


def validate_manifest(data: dict) -> None:
    for key in ("run_id", "source", "target_lang", "outputs"):
        if key not in data:
            raise ManifestError(f"missing key: {key}")
    if not data["outputs"]:
        raise ManifestError("no outputs listed")
    ids = set()
    for o in data["outputs"]:
        for k in ("output_id", "vendor", "file"):
            if k not in o:
                raise ManifestError(f"output missing {k}: {o}")
        if not NAME_RE.match(o["output_id"]):
            raise ManifestError(f"bad output_id (use lowercase, digits, _ or -): {o['output_id']}")
        if o["output_id"] in ids:
            raise ManifestError(f"duplicate output_id: {o['output_id']}")
        ids.add(o["output_id"])


def example_manifest(run_id: str = "20261030_131500_ms_clip01") -> dict:
    return {
        "run_id": run_id,
        "source": {"file": "src__en__clip01.mp4", "lang": "en"},
        "target_lang": "ms",
        "outputs": [
            {"output_id": "heygen__ms__clip01", "vendor": "HeyGen", "file": "heygen__ms__clip01.mp4"},
            {"output_id": "sync__ms__clip01", "vendor": "Sync.so", "file": "sync__ms__clip01.mp4"},
            {"output_id": "framex__ms__clip01", "vendor": "FrameX", "file": "framex__ms__clip01.mp4"},
        ],
    }
