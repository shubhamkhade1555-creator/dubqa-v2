"""Turn measured values into Good / Check / Fix labels and an overall score.

Missing values (None) are never treated as a pass. They are excluded and
reported through coverage.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_thresholds(path: Path | None = None) -> dict:
    p = path or ROOT / "config" / "thresholds.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def label(key: str, value, th: dict) -> str | None:
    if value is None:
        return None
    m = th["metrics"][key]
    d = m["direction"]
    if d == "min":
        return "Good" if value >= m["good"] else ("Check" if value >= m["check"] else "Fix")
    if d == "max":
        return "Good" if value <= m["good"] else ("Check" if value <= m["check"] else "Fix")
    if d == "max_abs":
        v = abs(value)
        return "Good" if v <= m["good"] else ("Check" if v <= m["check"] else "Fix")
    if d == "band":
        if m["low"] <= value <= m["high"]:
            return "Good"
        if m["check_low"] <= value <= m["check_high"]:
            return "Check"
        return "Fix"
    raise ValueError(f"unknown direction for {key}: {d}")


POINTS = {"Good": 1.0, "Check": 0.5, "Fix": 0.0}


def score_output(values: dict, th: dict) -> dict:
    """values: metric_key -> measured value (or None). Returns labels, group scores, overall."""
    labels = {k: label(k, values.get(k), th) for k in th["metrics"]}
    groups: dict[str, list[float]] = {}
    for k, lab in labels.items():
        if lab is None:
            continue
        g = th["metrics"][k]["group"]
        groups.setdefault(g, []).append(POINTS[lab])
    group_scores = {g: round(100.0 * sum(v) / len(v), 1) for g, v in groups.items()}
    weights = th["group_weights"]
    used = {g: weights[g] for g in group_scores if g in weights}
    total_w = sum(used.values())
    overall = round(sum(group_scores[g] * w for g, w in used.items()) / total_w, 1) if total_w else None
    measured = sum(1 for lab in labels.values() if lab is not None)
    coverage = round(100.0 * measured / len(labels), 1)
    fixes = [k for k, lab in labels.items() if lab == "Fix"]
    gate = th["release_gate"]
    critical_fail = any(k in ("missing_sentences", "hallucinations") and labels[k] == "Fix" for k in labels)
    approved = (overall is not None and overall >= gate["min_overall"]
                and coverage >= 100 - gate["max_missing_coverage_pct"]
                and not critical_fail and not fixes)
    return {"labels": labels, "group_scores": group_scores, "overall": overall,
            "coverage_pct": coverage, "fix_metrics": fixes,
            "release": "Approved" if approved else ("Review" if overall is not None else "Not measured")}


def rank_outputs(scores: dict[str, float | None]) -> dict[str, str]:
    """Competition ranking with ties (1, 1, 3). None ranks last with no rank."""
    ranked = sorted([(v, k) for k, v in scores.items() if v is not None], key=lambda x: -x[0])
    out, prev, pos = {}, None, 0
    for i, (v, k) in enumerate(ranked, start=1):
        if v != prev:
            pos = i
            prev = v
        tied = sum(1 for vv, _ in ranked if vv == v)
        out[k] = f"{pos} (tie)" if tied > 1 else str(pos)
    for k, v in scores.items():
        if v is None:
            out[k] = "—"
    return out
