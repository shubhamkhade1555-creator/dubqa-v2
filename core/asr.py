"""Word-level transcription and language detection with Gemini API.

Replaces faster-whisper with Google Gemini audio understanding.
Falls back cleanly to precomputed JSON files or reports errors if GEMINI_API_KEY is unset.
"""
import json
import os
import time
from pathlib import Path
from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    word: str
    start: float = Field(description="Start time in seconds with millisecond precision, e.g. 0.45")
    end: float = Field(description="End time in seconds with millisecond precision, e.g. 0.82")


class SentenceSegment(BaseModel):
    text: str
    start: float
    end: float


class AudioTranscriptionSchema(BaseModel):
    language: str = Field(description="Detected ISO 639-1 language code, e.g. 'en', 'es', 'hi'")
    language_prob: float = Field(description="Confidence probability between 0.0 and 1.0", default=1.0)
    words: list[WordTimestamp]
    sentences: list[SentenceSegment]


def transcribe_words(wav: Path, lang: str | None = None) -> dict:
    """Transcribes audio using Gemini and extracts word-level timestamps and detected language."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment or .env file.")

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("google-genai library is not installed. Run: pip install google-genai") from e

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # 1. Upload audio to Gemini File API
    audio_file = client.files.upload(file=str(wav))
    try:
        # Wait for file to become active
        for _ in range(30):
            status = client.files.get(name=audio_file.name).state.name
            if status == "ACTIVE":
                break
            if status == "FAILED":
                raise RuntimeError(f"Audio file processing failed on Gemini server: {audio_file.name}")
            time.sleep(1)

        # 2. Build structured transcription prompt
        prompt = (
            "You are an expert audio transcription system. Listen to the attached audio carefully.\n"
            "Generate an exact, highly accurate verbatim transcript with precise word-level start and end timestamps (in seconds).\n"
            "Ensure every spoken word has its corresponding start and end offset.\n"
        )
        if lang:
            prompt += f"Expected spoken language: {lang}.\n"

        response = client.models.generate_content(
            model=model_name,
            contents=[audio_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AudioTranscriptionSchema,
                temperature=0.0,
            ),
        )

        data = json.loads(response.text)

        # 3. Format dictionary to conform with DubQA pipeline requirements
        words = [
            {"word": w["word"].strip(), "start": round(float(w["start"]), 3), "end": round(float(w["end"]), 3)}
            for w in data.get("words", [])
        ]
        sentences = [
            {"text": s["text"].strip(), "start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3)}
            for s in data.get("sentences", [])
        ]

        return {
            "words": words,
            "sentences": sentences,
            "language": data.get("language", lang or "en"),
            "language_prob": round(float(data.get("language_prob", 1.0)), 3),
        }

    finally:
        # Clean up uploaded audio file from Gemini remote storage
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass


def load_words(path: Path) -> dict:
    """Loads precomputed transcription JSON file (used for QC mode)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
