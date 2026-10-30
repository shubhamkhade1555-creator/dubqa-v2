import numpy as np

from core import timing
from core.vad import in_any


def test_onset_and_tail_match_designed_offsets():
    t = np.arange(0, 3, 0.04)
    op = np.full_like(t, 0.02)
    # word 0.5..1.3 s; mouth opens at +0.10 and closes at end +0.03
    op[(t >= 0.60) & (t < 1.33)] = 0.5
    words = [{"word": "a", "start": 0.5, "end": 1.3}]
    thr = timing.speaker_threshold(op)
    d = timing.word_drifts(words, t, op, thr)[0]
    assert abs(d["onset_ms"] - 100) <= 40          # frame grid is 40 ms
    assert abs(d["tail_ms"] - 30) <= 40


def test_early_close_gives_negative_tail():
    t = np.arange(0, 3, 0.04)
    op = np.full_like(t, 0.02)
    op[(t >= 0.52) & (t < 1.28)] = 0.5             # closes 20 ms before word end
    words = [{"word": "a", "start": 0.5, "end": 1.3}]
    d = timing.word_drifts(words, t, op, timing.speaker_threshold(op))[0]
    assert d["tail_ms"] is not None and d["tail_ms"] < 0


def test_false_movement_only_counts_silence():
    t = np.arange(0, 2, 0.04)
    op = np.full_like(t, 0.02)
    speech = [(0.0, 1.0)]
    op[(t >= 1.2) & (t < 1.4)] = 0.5               # blink inside silence
    thr = timing.speaker_threshold(op)
    fm = timing.false_movement(t, op, thr, speech)
    assert fm is not None and 0 < fm < 50


def test_word_pass_counts_unmatched_as_fail():
    drifts = [{"onset_ms": 20, "tail_ms": 20}, {"onset_ms": None, "tail_ms": None}]
    assert timing.word_pass_rate(drifts) == 50.0


def test_in_any():
    assert in_any(0.5, [(0.0, 1.0)]) and not in_any(1.5, [(0.0, 1.0)])
