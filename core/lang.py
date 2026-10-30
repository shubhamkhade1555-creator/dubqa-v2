"""Language detection and resolution.

Order of evidence for each video:
1. Whisper's detected language (full mode), or a "language" field in the words JSON (precomputed).
2. Gemini's language field from the extraction (when Gemini ran).
3. Unicode script of the transcript words. This identifies non-Latin scripts reliably
   (Devanagari -> hi, Tamil -> ta, ...). Latin script cannot separate en, es, fr, de, so it
   is reported as undetermined and the declared value is kept.

A declared language in the manifest wins unless it is "auto" or missing. A disagreement
with the detected language is recorded as a note, never silently overwritten.
"""
from collections import Counter

# Script name, code, Unicode block. Devanagari covers hi, mr, ne: script alone cannot separate them.
SCRIPTS = [
    ("Devanagari", "hi", 0x0900, 0x097F),
    ("Bengali", "bn", 0x0980, 0x09FF),
    ("Gurmukhi", "pa", 0x0A00, 0x0A7F),
    ("Gujarati", "gu", 0x0A80, 0x0AFF),
    ("Tamil", "ta", 0x0B80, 0x0BFF),
    ("Telugu", "te", 0x0C00, 0x0C7F),
    ("Kannada", "kn", 0x0C80, 0x0CFF),
    ("Malayalam", "ml", 0x0D00, 0x0D7F),
    ("Thai", "th", 0x0E00, 0x0E7F),
    ("Arabic", "ar", 0x0600, 0x06FF),
    ("Greek", "el", 0x0370, 0x03FF),
    ("Cyrillic", "ru", 0x0400, 0x04FF),
    ("Hangul", "ko", 0xAC00, 0xD7AF),
    ("Kana", "ja", 0x3040, 0x30FF),
    ("Han", "zh", 0x4E00, 0x9FFF),
]
LATIN_RANGES = [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)]


def _is_latin(ch: str) -> bool:
    c = ord(ch)
    return any(lo <= c <= hi for lo, hi in LATIN_RANGES)


def detect_from_text(text: str) -> dict:
    counts: Counter = Counter()
    latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        c = ord(ch)
        hit = None
        for name, code, lo, hi in SCRIPTS:
            if lo <= c <= hi:
                hit = (name, code)
                break
        if hit:
            counts[hit] += 1
        elif _is_latin(ch):
            latin += 1
    total = sum(counts.values()) + latin
    if total == 0:
        return {"lang": None, "method": "script", "script": None, "confidence": 0.0,
                "note": "no letters to detect from"}
    if counts:
        (name, code), n = counts.most_common(1)[0]
        share = round(n / total, 3)
        if share >= 0.5:
            note = "Devanagari is shared by hi, mr and ne" if code == "hi" else ""
            return {"lang": code, "method": "script", "script": name, "confidence": share, "note": note}
    return {"lang": None, "method": "script", "script": "Latin", "confidence": round(latin / total, 3),
            "note": "Latin script: cannot tell English, Spanish, French or German from script alone"}


def resolve(declared: str | None, detected: str | None) -> tuple[str | None, list[str]]:
    """Return the language to use and any notes. Declared wins unless it is 'auto' or missing."""
    notes: list[str] = []
    decl = (declared or "").strip().lower()
    if decl and decl != "auto":
        if detected and detected != decl:
            notes.append(f"declared '{decl}' but detected '{detected}'; declared value used")
        return decl, notes
    if detected:
        return detected, notes
    notes.append("language undetermined; set it in the form if known")
    return None, notes


def detect_for_words(words: list[dict], language_field: str | None) -> dict:
    """Use an explicit language field when present, otherwise fall back to script detection."""
    if language_field:
        return {"lang": language_field.lower(), "method": "provided", "script": None,
                "confidence": 1.0, "note": ""}
    text = " ".join(w.get("word", "") for w in words)
    return detect_from_text(text)
