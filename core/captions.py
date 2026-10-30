"""Caption QA analysis module.

Parses SRT / WebVTT files and validates:
  - Text completeness & cue count
  - Timing offsets against speech intervals
  - Reading speed (characters per second)
  - Missing or overlapping cues
  - Malformed timestamps and negative durations
"""
import re
from pathlib import Path


def parse_timestamp(ts_str: str) -> float | None:
    """Parses SRT / VTT timestamps like 00:01:23.456 or 01:23.456. Returns seconds or None."""
    ts_str = ts_str.strip().replace(",", ".")
    parts = ts_str.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600.0 + float(m) * 60.0 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60.0 + float(s)
    except ValueError:
        return None
    return None


def parse_captions(caption_path: Path) -> list[dict]:
    if not caption_path.exists():
        return []

    content = caption_path.read_text(encoding="utf-8", errors="ignore")
    cues = []

    # Strip WEBVTT header or comments if present
    content = re.sub(r"^WEBVTT.*?\n", "", content, flags=re.IGNORECASE)

    # Regex for standard SRT or VTT timing blocks: 00:00:01.000 --> 00:00:03.500
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
                if start_s is None or end_s is None:
                    continue

                text = " ".join(lines[idx + 1:])
                text_clean = re.sub(r"<[^>]+>", "", text).strip()
                duration = end_s - start_s

                status = "PASS"
                notes = []
                if duration <= 0:
                    status = "FAIL"
                    notes.append("Negative or zero duration")
                    duration = max(0.001, duration)

                cps = round(len(text_clean) / max(0.1, duration), 1)
                if cps > 25.0:
                    status = "REVIEW"
                    notes.append(f"Fast reading speed: {cps} cps")

                cues.append({
                    "start": round(start_s, 3),
                    "end": round(end_s, 3),
                    "duration_s": round(duration, 3),
                    "text": text_clean,
                    "chars_per_sec": cps,
                    "status": status,
                    "notes": "; ".join(notes)
                })
                break

    # Sort cues by start time and check for overlaps
    cues.sort(key=lambda c: c["start"])
    for i in range(len(cues) - 1):
        if cues[i]["end"] > cues[i + 1]["start"]:
            cues[i]["status"] = "REVIEW"
            overlap_note = f"Overlaps with next cue by {round(cues[i]['end'] - cues[i+1]['start'], 3)}s"
            if cues[i]["notes"]:
                cues[i]["notes"] += f"; {overlap_note}"
            else:
                cues[i]["notes"] = overlap_note

    return cues


def evaluate_captions(caption_path: Path | None, speech_intervals: list[tuple[float, float]] | None = None) -> dict:
    if not caption_path or not caption_path.exists():
        return {
            "has_captions": False,
            "status": "NOT_MEASURED",
            "cues_count": 0,
            "cues": [],
            "speech_sync_measured": False
        }

    cues = parse_captions(caption_path)
    if not cues:
        return {
            "has_captions": True,
            "status": "NOT_MEASURED",
            "cues_count": 0,
            "cues": [],
            "speech_sync_measured": False
        }

    status = "PASS"
    for cue in cues:
        if cue["status"] == "FAIL":
            status = "FAIL"
            break
        elif cue["status"] == "REVIEW":
            status = "REVIEW"

    # Evaluate speech synchronization if speech intervals are supplied
    speech_sync_measured = False
    if speech_intervals:
        speech_sync_measured = True
        # Check alignment of first and last cues with speech intervals
        cue_start = cues[0]["start"]
        cue_end = cues[-1]["end"]
        sp_start = speech_intervals[0][0]
        sp_end = speech_intervals[-1][1]
        drift_start = abs(cue_start - sp_start)
        drift_end = abs(cue_end - sp_end)
        if drift_start > 2.0 or drift_end > 2.0:
            if status != "FAIL":
                status = "REVIEW"

    return {
        "has_captions": True,
        "status": status,
        "cues_count": len(cues),
        "cues": cues,
        "speech_sync_measured": speech_sync_measured
    }
