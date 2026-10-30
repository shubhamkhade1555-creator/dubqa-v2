import json
import pytest
from pathlib import Path
from core import audio_quality, captions, scenarios


def test_audio_quality_measurement_synthetic(tmp_path):
    from core import media
    import numpy as np
    import wave

    wav_p = tmp_path / "test.wav"
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Sine wave
    samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(wav_p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())

    res = audio_quality.measure_audio_quality(wav_p)
    assert res["duration_s"] == 1.0
    assert res["sample_rate"] == 16000
    assert res["clipped_sample_ratio"] == 0.0
    assert res["true_peak_dbtp"] < 0.0


def test_captions_parsing_and_evaluation(tmp_path):
    srt_p = tmp_path / "test.srt"
    srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello world test caption.

2
00:00:03,500 --> 00:00:05,000
Second subtitle line.
"""
    srt_p.write_text(srt_content, encoding="utf-8")
    res = captions.evaluate_captions(srt_p)
    assert res["has_captions"] is True
    assert res["cues_count"] == 2
    assert res["status"] == "PASS"


def test_scenario_evaluations_synthetic(tmp_path):
    outputs = {
        "out_1": {
            "values": {"onset_drift_ms": 30.0, "false_movement_pct": 5.0, "missing_sentences": 0},
            "extras": {"missing_words": []}
        }
    }
    aq = {"out_1": {"clipped_sample_ratio": 0.0}}
    cap = {"out_1": {"has_captions": False}}
    sc_results, defects = scenarios.run_scenario_evaluations(outputs, aq, cap, tmp_path)
    assert len(sc_results) > 0
    # Onset is within threshold, should PASS LS-02
    ls02 = next(s for s in sc_results if s["scenario_id"] == "LS-02")
    assert ls02["status"] == "PASS"


def test_scenario_detects_defect_on_failure(tmp_path):
    outputs = {
        "out_1": {
            "values": {"onset_drift_ms": 120.0, "false_movement_pct": 25.0, "missing_sentences": 1},
            "extras": {"missing_words": ["dropped_word"]}
        }
    }
    aq = {"out_1": {"clipped_sample_ratio": 0.05}}
    cap = {"out_1": {"has_captions": False}}
    sc_results, defects = scenarios.run_scenario_evaluations(outputs, aq, cap, tmp_path)
    # Check that critical/high defects were generated
    def_cats = [d["category"] for d in defects]
    assert "lipsync" in def_cats
    assert "completeness" in def_cats
    assert "audio" in def_cats
