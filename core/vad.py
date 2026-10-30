"""Energy-based voice activity detection (no model needed).

Frames of 20 ms. A frame is speech when its RMS is above an adaptive floor.
Returns merged speech intervals in seconds.
"""
import numpy as np


def speech_intervals(samples: np.ndarray, sr: int, frame_ms: int = 20,
                     rel_db: float = -30.0, min_len_s: float = 0.08,
                     merge_gap_s: float = 0.15) -> list[tuple[float, float]]:
    if samples.size == 0:
        return []
    n = int(sr * frame_ms / 1000)
    n_frames = samples.size // n
    if n_frames == 0:
        return []
    frames = samples[: n_frames * n].reshape(n_frames, n)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    ref = np.percentile(rms, 95)
    thr = ref * (10 ** (rel_db / 20.0))
    active = rms > thr
    intervals, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        if not a and start is not None:
            intervals.append([start, i]); start = None
    if start is not None:
        intervals.append([start, n_frames])
    dt = frame_ms / 1000.0
    secs = [(s * dt, e * dt) for s, e in intervals]
    # merge close intervals
    merged: list[list[float]] = []
    for s, e in secs:
        if merged and s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in merged if e - s >= min_len_s]


def in_any(t: float, intervals) -> bool:
    return any(s <= t < e for s, e in intervals)
