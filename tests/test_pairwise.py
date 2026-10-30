from core import pairwise
from core.scoring import load_thresholds

TH = load_thresholds()


def test_lower_is_better_metric():
    assert pairwise.better("timing_drift_ms", 200, 1200, TH) == "a"


def test_tie_within_tolerance():
    assert pairwise.better("translation_fidelity", 95, 96, TH) == "tie"


def test_band_metric_closest_to_one_wins():
    assert pairwise.better("syllable_ratio", 1.10, 1.30, TH) == "a"


def test_matrix_counts_pairs_once():
    out = pairwise.compare_all({
        "x": {"timing_drift_ms": 100, "onset_drift_ms": 50},
        "y": {"timing_drift_ms": 500, "onset_drift_ms": 90},
    }, TH)
    assert len(out["pairs"]) == 1
    assert out["win_matrix"]["x"]["y"] == 2
