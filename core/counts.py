"""Word and syllable counts by code. Syllables are a heuristic, labelled as such in reports.

Devanagari: each syllable has one vowel nucleus, so the count is
  independent vowels + vowel signs (matras) + consonants that carry the inherent 'a'
  (no vowel sign and no virama after them).
Latin script: vowel groups, with a common silent-e rule. This is approximate for English
and weaker for other Latin-script languages.
"""
import re
import unicodedata

DEV_INDEP_VOWEL = range(0x0904, 0x0915)   # अ..ऑ
DEV_CONSONANT = range(0x0915, 0x093A)     # क..ह
DEV_MATRA = range(0x093E, 0x094D)         # ा..ौ
VIRAMA = 0x094D
LATIN_VOWELS = re.compile(r"[aeiouy]+")


def tokens(text: str) -> list[str]:
    """Words are whitespace-separated. Punctuation and symbols are stripped, but combining marks
    (Devanagari vowel signs, virama) are kept. Python's \\w drops those marks and splits Hindi words."""
    out = []
    for tok in (text or "").split():
        clean = "".join(ch for ch in tok if not unicodedata.category(ch).startswith(("P", "S")))
        if any(unicodedata.category(ch)[0] in "LM" for ch in clean):
            out.append(clean)
    return out


def word_count(text: str) -> int:
    return len(tokens(text))


def _devanagari_nuclei(word: str) -> int:
    n = 0
    for i, ch in enumerate(word):
        c = ord(ch)
        if c in DEV_INDEP_VOWEL or c in DEV_MATRA:
            n += 1
        elif c in DEV_CONSONANT:
            nxt = ord(word[i + 1]) if i + 1 < len(word) else None
            if nxt == VIRAMA or (nxt is not None and nxt in DEV_MATRA):
                continue
            n += 1
    return n


def _latin_syllables(word: str) -> int:
    w = word.lower()
    groups = LATIN_VOWELS.findall(w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1 if re.search(r"[a-z]", w) else 0)


def syllable_count(text: str) -> int:
    total = 0
    for word in tokens(text):
        if any(0x0900 <= ord(ch) <= 0x097F for ch in word):
            total += max(_devanagari_nuclei(word), 1)
        else:
            total += _latin_syllables(word)
    return total
