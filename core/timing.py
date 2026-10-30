"""Timing metrics. Pure numpy so every function can be unit-tested without models.

Units: time in seconds in, milliseconds out where the name says _ms.
"""
import numpy as np

from core.vad import in_any

ONSET_WINDOW = (-0.30, 0.50)   # search window around word start
TAIL_WINDOW = (-0.20, 0.50)    # search from slightly before word end (early close = negative tail)


def speaker_threshold(opening: np.ndarray) -> float:
    valid = opening[~np.isnan(opening)]
    if valid.size == 0:
        return float("nan")
    return 0.20 * float(np.percentile(valid, 95))


def word_drifts(words: list[dict], t: np.ndarray, opening: np.ndarray, thr: float):
    """Per word: onset_ms, tail_ms (None when no mouth event found in the window)."""
    out = []
    for w in words:
        s, e = float(w["start"]), float(w["end"])
        onset = None
        m = (t >= s + ONSET_WINDOW[0]) & (t <= s + ONSET_WINDOW[1]) & (opening > thr)
        if m.any():
            onset = (t[np.argmax(m)] - s) * 1000.0
        tail = None
        m2 = (t >= e + TAIL_WINDOW[0]) & (t <= e + TAIL_WINDOW[1]) & (opening < thr)
        if m2.any():
            tail = (t[np.argmax(m2)] - e) * 1000.0
        out.append({"word": w.get("word", ""), "start": s, "end": e,
                    "onset_ms": None if onset is None else round(onset, 1),
                    "tail_ms": None if tail is None else round(tail, 1)})
    return out


def word_pass_rate(drifts: list[dict], good_ms: float = 80.0):
    scored = [d for d in drifts if d["onset_ms"] is not None and d["tail_ms"] is not None]
    if not drifts:
        return None
    ok = sum(1 for d in scored if abs(d["onset_ms"]) < good_ms and abs(d["tail_ms"]) < good_ms)
    return round(100.0 * ok / len(drifts), 1)   # unmatched words count as fails


def median_abs(values) -> float | None:
    vals = [abs(v) for v in values if v is not None]
    return round(float(np.median(vals)), 1) if vals else None


def open_ratio_during_speech(t, opening, thr, speech) -> float | None:
    mask = np.array([in_any(x, speech) for x in t])
    if not mask.any():
        return None
    return round(100.0 * float(np.mean(opening[mask] > thr)), 1)


def false_movement(t, opening, thr, speech) -> float | None:
    mask = np.array([not in_any(x, speech) for x in t])
    if not mask.any():
        return None
    return round(100.0 * float(np.mean(opening[mask] > thr)), 1)


def av_offset_ms(t_lips, opening, speech, max_lag_ms: float = 500.0) -> float | None:
    """Lag (ms) that best aligns the lip-open signal with the speech signal.
    Positive = mouth moves after the audio."""
    if t_lips.size < 4 or not speech:
        return None
    grid = np.arange(0, t_lips[-1], 0.01)
    if grid.size < 4:
        return None
    speech_sig = np.array([1.0 if in_any(x, speech) else 0.0 for x in grid])
    lip = np.interp(grid, t_lips, np.nan_to_num(opening, nan=0.0))
    a = speech_sig - speech_sig.mean()
    b = lip - lip.mean()
    lags = np.arange(-int(max_lag_ms / 10), int(max_lag_ms / 10) + 1)
    best, best_lag = -np.inf, 0
    for k in lags:
        if k >= 0:
            x, y = a[: a.size - k], b[k:]
        else:
            x, y = a[-k:], b[: b.size + k]
        if x.size == 0:
            continue
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
        if c > best:
            best, best_lag = c, k
    return float(best_lag * 10)


def speech_rate_ratio(dub_words: int, dub_dur: float, src_words: int, src_dur: float):
    if min(dub_dur, src_dur, src_words) <= 0:
        return None
    return round((dub_words / dub_dur) / (src_words / src_dur), 3)


def duration_diff_pct(dub_dur: float, src_dur: float) -> float | None:
    if src_dur <= 0:
        return None
    return round(100.0 * abs(dub_dur - src_dur) / src_dur, 2)
