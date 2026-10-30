import pytest

from core import align
from core.manifest import ManifestError, example_manifest, validate_manifest


def test_example_manifest_is_valid():
    validate_manifest(example_manifest())


def test_duplicate_output_id_rejected():
    m = example_manifest()
    m["outputs"].append(dict(m["outputs"][0]))
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_bad_output_id_rejected():
    m = example_manifest()
    m["outputs"][0]["output_id"] = "Bad Name!"
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_missing_word_detected():
    src = [{"word": w, "start": i, "end": i + 0.5} for i, w in enumerate(["a", "b", "c"])]
    dub = [{"word": w, "start": i, "end": i + 0.5} for i, w in enumerate(["a", "c"])]
    pairs, missing, extra = align.align_words(src, dub)
    assert missing == ["b"] and extra == []
    assert align.lexical_fidelity(src, pairs) == 66.7
