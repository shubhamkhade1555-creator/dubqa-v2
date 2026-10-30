"""Runs the whole precomputed pipeline on the synthetic batch with KNOWN timing errors."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core import pipeline
from report import build_html
from scripts import make_synthetic

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("qc") / "QC_synthetic"
    d.mkdir()
    make_synthetic.build(d)
    pipeline.run(d, mode="precomputed")
    build_html.build(d)
    return d


def test_results_written(run_dir):
    res = json.loads((run_dir / "results.json").read_text())
    assert set(res["outputs"]) == {"heygen__ms__clip01", "sync__ms__clip01", "framex__ms__clip01"}


def test_onset_matches_known_truth(run_dir):
    res = json.loads((run_dir / "results.json").read_text())["outputs"]
    assert abs(res["heygen__ms__clip01"]["values"]["onset_drift_ms"] - 100) <= 45
    assert abs(res["sync__ms__clip01"]["values"]["onset_drift_ms"] - 60) <= 45
    assert abs(res["framex__ms__clip01"]["values"]["onset_drift_ms"] - 20) <= 45


def test_framex_missing_word_detected(run_dir):
    res = json.loads((run_dir / "results.json").read_text())["outputs"]["framex__ms__clip01"]
    assert res["extras"]["missing_words"] == ["w2"]


def test_pairs_cover_every_dub_pair(run_dir):
    res = json.loads((run_dir / "results.json").read_text())
    assert len(res["pairs"]) == 3


def test_report_built(run_dir):
    html = (run_dir / "report" / "index.html").read_text()
    assert "Pairwise comparison" in html and "Synchronised players" in html
