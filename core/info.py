"""Gemini extraction (labels and transcripts) plus code-based counts and segment matching.

Gemini returns: language, speakers, and segments with approximate start/end, speaker, emotion,
tone and transcript. Code then computes word counts, syllable counts, segment counts, speaker
counts and emotion agreement from those fields.

Gemini's start/end values are approximate. They are used only to pair source and dub segments
and to label the report. They are never used for lip-sync or timing numbers, which come from
the word and lip measurements.
"""
import json
import re
from pathlib import Path

from core import counts, judge_gemini

EXTRACT_PROMPT = """You are labelling one video for dubbing QA. Watch and listen to it.
Return ONLY JSON with this shape:
{"language": "<ISO 639-1 code of the spoken language>",
 "speakers": [{"id": "S1", "description": "<short description>"}],
 "segments": [{"start": 0.0, "end": 0.0, "speaker": "S1",
               "emotion": "<one word, e.g. joy, anger, sadness, neutral, fear, surprise>",
               "tone": "<short phrase, e.g. calm, urgent, sarcastic>",
               "transcript": "<exact spoken words in this segment>"}]}
Rules:
- Segments are in spoken order. start and end are seconds from the start of the video.
- transcript is what is actually spoken in that language. Do not translate. Do not invent speech.
- If nothing is spoken, return empty speakers and segments lists.
"""


class ExtractionError(ValueError):
    pass


def validate_extraction(obj: dict) -> dict:
    """Keep only well-formed segments. Count what was dropped so the report can say so."""
    if not isinstance(obj, dict):
        raise ExtractionError("top level is not an object")
    raw = obj.get("segments")
    if not isinstance(raw, list):
        raise ExtractionError("segments is not a list")
    clean, dropped = [], 0
    for s in raw:
        try:
            start, end = float(s["start"]), float(s["end"])
            if not (end > start >= 0):
                raise ValueError("bad times")
            clean.append({
                "start": round(start, 2), "end": round(end, 2),
                "speaker": str(s.get("speaker") or "S1"),
                "emotion": str(s.get("emotion") or "").strip().lower(),
                "tone": str(s.get("tone") or "").strip(),
                "transcript": str(s.get("transcript") or "").strip(),
            })
        except (KeyError, TypeError, ValueError):
            dropped += 1
    clean.sort(key=lambda x: x["start"])
    speakers = obj.get("speakers") if isinstance(obj.get("speakers"), list) else []
    return {"language": str(obj.get("language") or "").strip().lower() or None,
            "speakers": speakers, "segments": clean, "dropped_segments": dropped}


def extract_video(media_path: Path, cache_path: Path, use_gemini: bool) -> dict:
    """One Gemini call per video. Cached as JSON so a rerun does not pay for the call again."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("status") == "ok":
            return cached
    if not use_gemini:
        return {"status": "skipped", "reason": "Gemini extraction turned off"}
    ok, reason = judge_gemini.gemini_available()
    if not ok:
        return {"status": "skipped", "reason": reason}
    try:
        raw = judge_gemini.call_gemini_json(media_path, EXTRACT_PROMPT)
    except Exception as e:  # network or quota errors must not fail the run
        return {"status": "error", "reason": str(e)[:300]}
    if raw.get("status") != "ok":
        return {"status": raw.get("status", "error"), "reason": raw.get("reason") or raw.get("raw", "")[:300]}
    try:
        out = validate_extraction(raw)
    except ExtractionError as e:
        return {"status": "parse_error", "reason": str(e)}
    out["status"] = "ok"
    out["model"] = raw.get("model")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def match_segments(src: list[dict], dub: list[dict]) -> tuple[list[tuple[int, int | None]], list[int]]:
    """Pair each source segment with the dub segment it overlaps most.

    Returns (pairs, unmatched_dub_indices). A source segment with no overlap gets None.
    Each dub segment is used at most once.
    """
    used: set[int] = set()
    pairs = []
    for i, s in enumerate(src):
        best, best_ov = None, 0.0
        for j, d in enumerate(dub):
            if j in used:
                continue
            ov = _overlap(s, d)
            if ov > best_ov:
                best, best_ov = j, ov
        if best is not None:
            used.add(best)
        pairs.append((i, best))
    unmatched = [j for j in range(len(dub)) if j not in used]
    return pairs, unmatched


def _pct_diff(base: float, other: float) -> float | None:
    if not base:
        return None
    return round(100.0 * (other - base) / base, 1)


def summarize(src_ext: dict | None, dub_ext: dict | None) -> dict:
    """Counts and per-segment rows. Returns status 'not measured' when Gemini did not run."""
    if not (src_ext and dub_ext and src_ext.get("status") == "ok" and dub_ext.get("status") == "ok"):
        return {"status": "not measured", "reason": _reason(src_ext, dub_ext)}
    s_segs, d_segs = src_ext["segments"], dub_ext["segments"]
    pairs, unmatched = match_segments(s_segs, d_segs)
    rows = []
    em_flags = []
    for i, j in pairs:
        s = s_segs[i]
        d = d_segs[j] if j is not None else None
        sw = counts.word_count(s["transcript"])
        ss = counts.syllable_count(s["transcript"])
        dw = counts.word_count(d["transcript"]) if d else None
        ds = counts.syllable_count(d["transcript"]) if d else None
        same_emotion = None
        if d and s["emotion"] and d["emotion"]:
            same_emotion = s["emotion"] == d["emotion"]
            em_flags.append(same_emotion)
        rows.append({
            "seg": i + 1, "src_time": f"{s['start']:.2f}-{s['end']:.2f}",
            "dub_time": f"{d['start']:.2f}-{d['end']:.2f}" if d else "no match",
            "src_text": s["transcript"], "dub_text": d["transcript"] if d else "",
            "src_speaker": s["speaker"], "dub_speaker": d["speaker"] if d else "",
            "src_emotion": s["emotion"], "dub_emotion": d["emotion"] if d else "",
            "src_words": sw, "dub_words": dw, "word_diff_pct": _pct_diff(sw, dw) if dw is not None else None,
            "src_syll": ss, "dub_syll": ds, "syll_diff_pct": _pct_diff(ss, ds) if ds is not None else None,
            "same_emotion": same_emotion,
        })
    sw_total = sum(counts.word_count(s["transcript"]) for s in s_segs)
    dw_total = sum(counts.word_count(d["transcript"]) for d in d_segs)
    ss_total = sum(counts.syllable_count(s["transcript"]) for s in s_segs)
    ds_total = sum(counts.syllable_count(d["transcript"]) for d in d_segs)
    return {
        "status": "ok",
        "src_language": src_ext.get("language"), "dub_language": dub_ext.get("language"),
        "segments_src": len(s_segs), "segments_dub": len(d_segs),
        "segment_count_diff": len(d_segs) - len(s_segs),
        "unmatched_dub_segments": len(unmatched),
        "speakers_src": len({s["speaker"] for s in s_segs}),
        "speakers_dub": len({d["speaker"] for d in d_segs}),
        "words_src": sw_total, "words_dub": dw_total,
        "word_diff_pct": _pct_diff(sw_total, dw_total),
        "syll_src": ss_total, "syll_dub": ds_total,
        "syll_diff_pct": _pct_diff(ss_total, ds_total),
        "emotion_match_pct": round(100.0 * sum(em_flags) / len(em_flags), 1) if em_flags else None,
        "dropped_segments": src_ext.get("dropped_segments", 0) + dub_ext.get("dropped_segments", 0),
        "rows": rows,
        "note": "Syllables are a heuristic. Segment times are Gemini's approximations.",
    }


def _reason(src_ext, dub_ext) -> str:
    for e in (src_ext, dub_ext):
        if e and e.get("status") != "ok":
            return f"{e.get('status')}: {e.get('reason', '')}".strip(": ")
    return "not run"


def checklist(run_dir: Path, man: dict, mode: str, use_gemini: bool) -> list[dict]:
    """Required data per run, with what is present and what is missing."""
    pre = run_dir / "precomputed"
    items = []

    def add(name, ok, detail=""):
        items.append({"item": name, "status": "present" if ok else "missing", "detail": detail})

    src = man["source"]["file"]
    add("source video", (run_dir / "inputs" / src).exists(), src)
    if mode == "precomputed":
        add("source words JSON", (pre / "source_words.json").exists())
        add("source lips JSON", (pre / "source_lips.json").exists())
    for o in man["outputs"]:
        oid = o["output_id"]
        add(f"{o['vendor']} video", (run_dir / "inputs" / o["file"]).exists(), o["file"])
        if mode == "precomputed":
            add(f"{o['vendor']} words JSON", (pre / f"{oid}_words.json").exists())
            add(f"{o['vendor']} lips JSON", (pre / f"{oid}_lips.json").exists())
    key_ok, reason = judge_gemini.gemini_available()
    add("Gemini key (for language, speakers, emotions, segments)", key_ok and use_gemini,
        "on" if key_ok and use_gemini else reason if not key_ok else "turned off in form")
    return items
