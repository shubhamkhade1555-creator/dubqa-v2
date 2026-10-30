"""Language detection, counts, Gemini extraction handling, segment matching, and a full run with a fake Gemini."""
import json
import shutil
from pathlib import Path

import pytest

from core import counts, info, judge_gemini, lang

ROOT = Path(__file__).resolve().parent.parent


# ---- language -------------------------------------------------------------

def test_devanagari_detected_as_hindi():
    d = lang.detect_from_text("नमस्ते दुनिया यह एक परीक्षण है")
    assert d["lang"] == "hi" and d["script"] == "Devanagari"


def test_latin_text_is_undetermined():
    d = lang.detect_from_text("hello world this is a test")
    assert d["lang"] is None and d["script"] == "Latin"


def test_declared_language_wins_and_notes_disagreement():
    code, notes = lang.resolve("en", "hi")
    assert code == "en" and notes


def test_auto_uses_detected():
    assert lang.resolve("auto", "hi") == ("hi", [])


def test_auto_without_detection_is_undetermined():
    code, notes = lang.resolve("auto", None)
    assert code is None and notes


def test_explicit_language_field_is_used():
    d = lang.detect_for_words([{"word": "w0"}], "ta")
    assert d["lang"] == "ta" and d["method"] == "provided"


# ---- counts ---------------------------------------------------------------

def test_word_count_latin_and_devanagari():
    assert counts.word_count("hello, world! नमस्ते") == 3


def test_syllables_devanagari():
    assert counts.syllable_count("नमस्ते") == 3      # na-ma-ste
    assert counts.syllable_count("हिन्दी") == 2       # hin-di


def test_syllables_latin_with_silent_e():
    assert counts.syllable_count("hello") == 2
    assert counts.syllable_count("world") == 1
    assert counts.syllable_count("make") == 1


# ---- Gemini output validation ---------------------------------------------

def test_validate_drops_bad_segments_and_counts_them():
    obj = {"language": "EN", "speakers": [],
           "segments": [{"start": 1.0, "end": 0.5, "transcript": "bad times"},
                        {"start": 0.0, "end": 1.0, "speaker": "S1", "emotion": "Joy", "transcript": "hi"}]}
    out = info.validate_extraction(obj)
    assert out["language"] == "en"
    assert len(out["segments"]) == 1 and out["dropped_segments"] == 1
    assert out["segments"][0]["emotion"] == "joy"


def test_validate_rejects_non_list_segments():
    with pytest.raises(info.ExtractionError):
        info.validate_extraction({"segments": "nope"})


def test_extract_skipped_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = info.extract_video(tmp_path / "x.mp4", tmp_path / "cache.json", use_gemini=True)
    assert out["status"] == "skipped"


def test_extract_turned_off(tmp_path):
    out = info.extract_video(tmp_path / "x.mp4", tmp_path / "cache.json", use_gemini=False)
    assert out["status"] == "skipped"


# ---- segment matching and summary -----------------------------------------

def _seg(s, e, text, emo="neutral", spk="S1"):
    return {"start": s, "end": e, "transcript": text, "emotion": emo, "speaker": spk}


def test_match_by_overlap_and_unmatched_dub():
    src = [_seg(0, 1, "one"), _seg(1, 2, "two")]
    dub = [_seg(0.1, 0.9, "uno"), _seg(1.2, 1.9, "dos"), _seg(5, 6, "extra")]
    pairs, unmatched = info.match_segments(src, dub)
    assert pairs == [(0, 0), (1, 1)] and unmatched == [2]


def test_summary_counts_and_emotion_agreement():
    src = {"status": "ok", "language": "en", "segments": [_seg(0, 1, "good morning everyone", "joy")]}
    dub = {"status": "ok", "language": "hi", "segments": [_seg(0, 1, "नमस्ते दुनिया", "sadness", "S1")]}
    s = info.summarize(src, dub)
    assert s["status"] == "ok"
    assert s["segments_src"] == 1 and s["segments_dub"] == 1
    assert s["words_src"] == 3 and s["words_dub"] == 2
    assert s["word_diff_pct"] == -33.3
    assert s["emotion_match_pct"] == 0.0


def test_summary_not_measured_without_gemini():
    s = info.summarize({"status": "skipped", "reason": "no key"}, {"status": "skipped", "reason": "no key"})
    assert s["status"] == "not measured"


def test_judge_prompt_still_forbids_timing_claims():
    p = judge_gemini.build_prompt("a", "b", {})
    assert "Do not estimate timing" in p


# ---- full run with a fake Gemini ------------------------------------------

def _fake_gemini(media_path, prompt):
    """Stands in for the Gemini call: returns the QC transcripts as labelled segments."""
    pre = Path(media_path).parent.parent / "precomputed"
    if Path(media_path).name.startswith("src__"):
        words_file = pre / "source_words.json"
    else:
        words_file = pre / f"{Path(media_path).stem}_words.json"
    data = json.loads(words_file.read_text(encoding="utf-8"))
    segs = [{"start": s["start"], "end": s["end"], "speaker": "S1", "emotion": "neutral",
             "tone": "flat", "transcript": s["text"]} for s in data["sentences"]]
    return {"status": "ok", "model": "fake", "language": "en", "speakers": [], "segments": segs}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_full_run_with_fake_gemini_builds_report(tmp_path, monkeypatch):
    from core import pipeline
    from report import build_html

    run = tmp_path / "run"
    shutil.copytree(ROOT / "runs" / "QC_synthetic", run,
                    ignore=shutil.ignore_patterns("report", "work", "measure", "judge", "gemini", "results.json", "run_meta.json"))
    for d in ("work", "measure", "judge", "report"):
        (run / d).mkdir(exist_ok=True)

    monkeypatch.setattr(judge_gemini, "gemini_available", lambda: (True, "ok"))
    monkeypatch.setattr(judge_gemini, "call_gemini_json", _fake_gemini)

    res = pipeline.run(run, mode="precomputed", use_judge=False, use_gemini=True)
    assert res["gemini_source"]["status"] == "ok"
    for oid, o in res["info"].items():
        assert o["summary"]["status"] == "ok"
        assert o["summary"]["segment_count_diff"] == 0
        assert o["summary"]["emotion_match_pct"] == 100.0
        assert res["outputs"][oid]["values"]["emotion_match_pct"] == 100.0
    assert any(c["item"].startswith("Gemini key") for c in res["required_data"])

    html = build_html.build(run).read_text(encoding="utf-8")
    assert "Segment detail" in html and "Required data checklist" in html
