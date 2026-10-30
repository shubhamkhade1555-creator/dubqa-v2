"""Gemini calls. One shared upload-and-generate helper, used by the extractor and the judge.

Timing numbers are never produced here. Gemini returns labels and transcripts; code measures.
Skipped cleanly when GEMINI_API_KEY or google-genai is missing.
"""
import json
import os
import time
from pathlib import Path

SCHEMA_HINT = """Return ONLY JSON with this shape:
{"emotion_src": "", "emotion_dub": "", "emotion_match": true,
 "tone_src": "", "tone_dub": "", "tone_match": true,
 "meaning_preserved": true,
 "issues": [{"time_s": 0.0, "type": "", "note": ""}],
 "summary": ""}"""


def build_prompt(seg_text_src: str, seg_text_dub: str, measured: dict) -> str:
    return (
        "You are a dubbing QA reviewer. Watch and listen to the clip.\n"
        f"Source text: {seg_text_src}\nDubbed text: {seg_text_dub}\n"
        f"Measured facts (do not change these): {json.dumps(measured)}\n"
        "Judge emotion, tone and meaning only. Do not estimate timing or lip-sync.\n"
        + SCHEMA_HINT
    )


def gemini_available() -> tuple[bool, str]:
    if not os.environ.get("GEMINI_API_KEY"):
        return False, "GEMINI_API_KEY not set"
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False, "google-genai not installed"
    return True, "ok"


def call_gemini_json(media_path: Path, prompt: str) -> dict:
    """Upload a video or audio file, ask for JSON, return the parsed object plus status."""
    ok, reason = gemini_available()
    if not ok:
        return {"status": "skipped", "reason": reason}
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    f = client.files.upload(file=str(media_path))
    for _ in range(60):
        if client.files.get(name=f.name).state.name == "ACTIVE":
            break
        time.sleep(2)
    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    resp = client.models.generate_content(
        model=model, contents=[f, prompt],
        config={"response_mime_type": "application/json"})
    try:
        out = json.loads(resp.text)
    except (json.JSONDecodeError, TypeError):
        return {"status": "parse_error", "raw": str(resp.text)[:500], "model": model}
    if not isinstance(out, dict):
        return {"status": "parse_error", "raw": str(resp.text)[:500], "model": model}
    out["status"] = "ok"
    out["model"] = model
    return out


def judge_clip(clip: Path, prompt: str) -> dict:
    return call_gemini_json(clip, prompt)
