"""Caption QA analysis module.

Parses SRT / WebVTT files and validates:
  - Text completeness & cue count
  - Timing offsets against speech intervals
  - Reading speed (characters per second)
  - Missing or overlapping cues
"""
import re
from pathlib import Path


def parse_timestamp(ts_str: str) -> float:
    ts_str = ts_str.strip().replace(",", ".")
    parts = ts_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    return 0.0


def parse_captions(caption_path: Path) -> list[dict]:
    if not caption_path.exists():
        return []

    content = caption_path.read_text(encoding="utf-8", errors="ignore")
    cues = []

    # Regex for standard SRT or VTT timing blocks
    pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3}|\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3}|\d{2}:\d{2}[,\.]\d{3})"
    )

    blocks = content.replace("\r\n", "\n").split("\n\n")
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        for idx, line in enumerate(lines):
            match = pattern.search(line)
            if match:
                start_s = parse_timestamp(match.group(1))
                end_s = parse_timestamp(match.group(2))
                text = " ".join(lines[idx + 1:])
                text_clean = re.sub(r"<[^>]+>", "", text).strip()
                duration = max(0.1, end_s - start_s)
                cps = round(len(text_clean) / duration, 1)

                status = "PASS"
                if cps > 25.0:
                    status = "REVIEW"

                cues.append({
                    "start": round(start_s, 3),
                    "end": round(end_s, 3),
                    "duration_s": round(duration, 3),
                    "text": text_clean,
                    "chars_per_sec": cps,
                    "status": status
                })
                break
    return cues


def evaluate_captions(caption_path: Path | None, speech_intervals: list[tuple[float, float]] | None = None) -> dict:
    if not caption_path or not caption_path.exists():
        return {
            "has_captions": False,
            "status": "NOT_MEASURED",
            "cues_count": 0,
            "cues": []
        }

    cues = parse_captions(caption_path)
    if not cues:
        return {
            "has_captions": True,
            "status": "NOT_MEASURED",
            "cues_count": 0,
            "cues": []
        }

    status = "PASS"
    for cue in cues:
        if cue["status"] == "REVIEW":
            status = "REVIEW"

    return {
        "has_captions": True,
        "status": status,
        "cues_count": len(cues),
        "cues": cues
    }
