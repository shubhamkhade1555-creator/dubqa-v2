from core import scoring

TH = scoring.load_thresholds()


def test_labels_min_max_band():
    assert scoring.label("translation_fidelity", 90, TH) == "Good"
    assert scoring.label("translation_fidelity", 80, TH) == "Check"
    assert scoring.label("translation_fidelity", 60, TH) == "Fix"
    assert scoring.label("timing_drift_ms", 150, TH) == "Good"
    assert scoring.label("syllable_ratio", 1.2, TH) == "Check"
    assert scoring.label("syllable_ratio", 1.4, TH) == "Fix"


def test_none_is_not_a_pass():
    assert scoring.label("false_movement_pct", None, TH) is None
    res = scoring.score_output({"false_movement_pct": None}, TH)
    assert res["labels"]["false_movement_pct"] is None
    assert res["coverage_pct"] < 100


def test_max_abs_uses_absolute_value():
    assert scoring.label("tail_drift_ms", -40, TH) == "Good"


def test_ranks_with_ties():
    r = scoring.rank_outputs({"a": 77, "b": 77, "c": 72, "d": None})
    assert r["a"] == "1 (tie)" and r["b"] == "1 (tie)"
    assert r["c"] == "3" and r["d"] == "—"
