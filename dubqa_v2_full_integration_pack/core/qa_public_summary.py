"""Create a deliberately limited public summary. Never exports media or raw transcripts."""
from pathlib import Path
import json

ALLOWED_OUTPUT_FIELDS = ("output_id", "vendor", "overall_rank")
ALLOWED_METRICS = ("translation_fidelity", "timing_drift_ms", "onset_drift_ms",
                   "tail_drift_ms", "word_pass_rate_pct", "false_movement_pct")

def build_sanitized_summary(results: dict, approved: bool = False) -> dict:
    if not approved:
        raise PermissionError("Public summary requires explicit approval=True")
    outputs = []
    for oid, item in (results.get("outputs") or {}).items():
        vals = item.get("values") or {}
        outputs.append({
            "output_id": oid,
            "vendor": item.get("vendor", "Unknown"),
            "rank": item.get("overall_rank"),
            "metrics": {k: vals.get(k, "NOT_MEASURED") for k in ALLOWED_METRICS},
        })
    return {
        "run_alias": "run-" + str(results.get("run_id", "unknown"))[-8:],
        "generated_date": str(results.get("generated", ""))[:10],
        "status": results.get("overall_status", "REVIEW"),
        "outputs": outputs,
        "scenario_counts": results.get("scenario_counts", {}),
        "defect_counts": results.get("defect_counts", {}),
        "privacy_notice": "Sanitized summary only. No media, transcripts, face crops or secrets.",
    }

def write_public_summary(results: dict, run_dir: str | Path, approved: bool = False) -> Path:
    data = build_sanitized_summary(results, approved=approved)
    dest = Path(run_dir) / "report" / "public" / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest
