"""Word and sentence alignment between source and dub transcripts."""
import difflib
import unicodedata


def norm(word: str) -> str:
    """Lowercase, punctuation and symbols removed. Combining marks are kept, so Hindi words stay whole."""
    return "".join(ch for ch in word if not unicodedata.category(ch).startswith(("P", "S"))).lower()


def align_words(src_words: list[dict], dub_words: list[dict]):
    """Return matched pairs (src_idx, dub_idx) and lists of missing/extra source words."""
    a = [norm(w["word"]) for w in src_words]
    b = [norm(w["word"]) for w in dub_words]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pairs, matched_src = [], set()
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            pairs.append((block.a + k, block.b + k))
            matched_src.add(block.a + k)
    missing = [src_words[i]["word"] for i in range(len(a)) if i not in matched_src]
    matched_dub = {j for _, j in pairs}
    extra = [dub_words[j]["word"] for j in range(len(b)) if j not in matched_dub]
    return pairs, missing, extra


def lexical_fidelity(src_words, pairs) -> float | None:
    if not src_words:
        return None
    return round(100.0 * len(pairs) / len(src_words), 1)


def missing_sentences(sentences: list[dict], src_words, pairs) -> int:
    """A source sentence is missing when less than half of its words are matched."""
    matched = {i for i, _ in pairs}
    missing = 0
    for s in sentences:
        idx = [i for i, w in enumerate(src_words) if s["start"] <= w["start"] < s["end"] + 1e-6]
        if idx and sum(1 for i in idx if i in matched) / len(idx) < 0.5:
            missing += 1
    return missing
