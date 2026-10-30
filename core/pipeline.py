"""End-to-end run for one batch.

mode="precomputed": uses word and lip JSON files under <run>/precomputed/ (QC and offline).
mode="full":        uses faster-whisper and MediaPipe on the real media (needs requirements-ml.txt).
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core import align, asr, judge_gemini, lang, lips, media, pairwise, scoring, timing, vad
from core import info as ginfo
from core.manifest import load_manifest

log = logging.getLogger("dubqa")


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _status(run_dir: Path, status: str, **extra) -> None:
    meta_p = run_dir / "run_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    meta.update({"status": status, "updated": datetime.now(timezone.utc).isoformat()}, **extra)
    _write(meta_p, meta)


def _words_and_lips(mode, run_dir, wav, video, pre_words, pre_lips, lang):
    if mode == "precomputed":
        words = asr.load_words(pre_words)
        t, op = lips.load_precomputed(pre_lips)
    else:
        words = asr.transcribe_words(wav, lang)
        t, op = lips.track_video_mediapipe(video)
    return words, t, op


def run(run_dir: Path, mode: str = "precomputed", use_judge: bool = False, use_gemini: bool = True) -> dict:
    run_dir = Path(run_dir)
    man = load_manifest(run_dir / "manifest.json")
    th = scoring.load_thresholds()
    inputs, work, pre = run_dir / "inputs", run_dir / "work", run_dir / "precomputed"
    _status(run_dir, "running", mode=mode)
    log.info("run %s started (mode=%s, judge=%s, gemini=%s)", man["run_id"], mode, use_judge, use_gemini)
    try:
        log.info("source: %s", man["source"]["file"])
        src_path = inputs / man["source"]["file"]
        src_info = media.probe(src_path)
        src_wav = media.extract_audio_wav(src_path, work / "source.wav")
        src_words, t_s, op_s = _words_and_lips(
            mode, run_dir, src_wav, src_path,
            pre / "source_words.json", pre / "source_lips.json", man["source"].get("lang"))
        # Persist extracted words and lips for full auditability
        _write(run_dir / "words" / "source_words.json", src_words)
        _write(run_dir / "lips" / "source_lips.json", {"t": t_s.tolist() if hasattr(t_s, "tolist") else list(t_s),
                                                       "opening": op_s.tolist() if hasattr(op_s, "tolist") else list(op_s)})
        src_w = src_words["words"]
        src_sent = src_words.get("sentences", [])
        src_speech_s = _speech_total(src_wav)

        src_det = lang.detect_for_words(src_w, src_words.get("language"))
        src_ext = ginfo.extract_video(src_path, run_dir / "gemini" / "source.json", use_gemini)
        if src_det["lang"] is None and src_ext.get("language"):
            src_det = {"lang": src_ext["language"], "method": "gemini", "script": None, "confidence": 1.0, "note": ""}
        src_lang, src_lang_notes = lang.resolve(man["source"].get("lang"), src_det["lang"])
        log.info("source language: %s (declared=%s, detected=%s via %s)", src_lang,
                 man["source"].get("lang"), src_det["lang"], src_det["method"])

        results = {"run_id": man["run_id"], "mode": mode, "generated": datetime.now(timezone.utc).isoformat(),
                   "thresholds_version": th["version"], "source": {**src_info, "file": src_path.name,
                   "words": len(src_w), "speech_s": src_speech_s, "language": src_lang},
                   "outputs": {}, "segments": [], "pairs": {}, "notes": [],
                   "languages": {"source": {"declared": man["source"].get("lang"), "resolved": src_lang,
                                            "detection": src_det, "notes": src_lang_notes}},
                   "gemini_source": _gemini_brief(src_ext), "info": {},
                   "required_data": ginfo.checklist(run_dir, man, mode, use_gemini)}
        # segments = source sentences (shared by every output)
        for i, s in enumerate(src_sent, start=1):
            results["segments"].append({"seg": i, "start": s["start"], "end": s["end"], "text": s["text"]})

        values_by_out: dict[str, dict] = {}
        for o in man["outputs"]:
            oid = o["output_id"]
            path = inputs / o["file"]
            log.info("measuring %s (%s)", oid, o["vendor"])
            info = media.probe(path)
            wav = media.extract_audio_wav(path, work / f"{oid}.wav")
            samples, sr = media.read_wav_mono(wav)
            speech = vad.speech_intervals(samples, sr)
            dwords, t, op = _words_and_lips(mode, run_dir, wav, path, pre / f"{oid}_words.json",
                                            pre / f"{oid}_lips.json", man.get("target_lang"))
            # Persist extracted words and lips locally for each dub output
            _write(run_dir / "words" / f"{oid}_words.json", dwords)
            _write(run_dir / "lips" / f"{oid}_lips.json", {"t": t.tolist() if hasattr(t, "tolist") else list(t),
                                                           "opening": op.tolist() if hasattr(op, "tolist") else list(op)})
            d_w = dwords["words"]
            dub_det = lang.detect_for_words(d_w, dwords.get("language"))
            dub_ext = ginfo.extract_video(path, run_dir / "gemini" / f"{oid}.json", use_gemini)
            if dub_det["lang"] is None and dub_ext.get("language"):
                dub_det = {"lang": dub_ext["language"], "method": "gemini", "script": None, "confidence": 1.0, "note": ""}
            dub_lang, dub_lang_notes = lang.resolve(man.get("target_lang"), dub_det["lang"])
            log.info("%s language: %s (declared=%s, detected=%s via %s)", oid, dub_lang,
                     man.get("target_lang"), dub_det["lang"], dub_det["method"])
            gem_summary = ginfo.summarize(src_ext, dub_ext)
            thr = timing.speaker_threshold(op)
            drifts = timing.word_drifts(d_w, t, op, thr)
            pairs, missing, extra = align.align_words(src_w, d_w)

            starts = [abs(d_w[j]["start"] - src_w[i]["start"]) * 1000 for i, j in pairs]
            dub_speech = _speech_total(wav)
            values = {
                "translation_fidelity": align.lexical_fidelity(src_w, pairs),
                "word_count_diff": round(100.0 * abs(len(d_w) - len(src_w)) / max(len(src_w), 1), 1),
                "syllable_ratio": round(dub_speech / src_speech_s, 3) if src_speech_s else None,
                "timing_drift_ms": round(float(np.median(starts)), 1) if starts else None,
                "onset_drift_ms": timing.median_abs([d["onset_ms"] for d in drifts]),
                "tail_drift_ms": timing.median_abs([d["tail_ms"] for d in drifts]),
                "open_ratio_pct": timing.open_ratio_during_speech(t, op, thr, speech),
                "false_movement_pct": timing.false_movement(t, op, thr, speech),
                "word_pass_rate_pct": timing.word_pass_rate(drifts),
                "missing_sentences": align.missing_sentences(src_sent, src_w, pairs) if src_sent else None,
                "hallucinations": None,
                "emotion_match_pct": None,
                "tone_match_pct": None,
                "audio_quality": None,
            }
            extras = {"duration_s": info["duration_s"],
                      "duration_diff_pct": timing.duration_diff_pct(info["duration_s"], src_info["duration_s"]),
                      "wpm_ratio": timing.speech_rate_ratio(len(d_w), info["duration_s"], len(src_w),
                                                           src_info["duration_s"]),
                      "av_offset_ms": timing.av_offset_ms(t, op, speech),
                      "missing_words": missing[:50], "extra_words": extra[:50],
                      "translation_method": "lexical-proxy" if mode == "precomputed" else "lexical-proxy (COMET not wired yet)",
                      "speaker_threshold": None if np.isnan(thr) else round(float(thr), 4)}

            if use_judge and os.environ.get("GEMINI_API_KEY"):
                judged = _judge_outputs(run_dir, path, oid, results["segments"], d_w, src_w, values, extras)
                values.update(judged["values"])
                extras["judge_notes"] = judged["notes"]

            if values.get("emotion_match_pct") is None and gem_summary.get("emotion_match_pct") is not None:
                values["emotion_match_pct"] = gem_summary["emotion_match_pct"]
                extras["emotion_source"] = "Gemini labels, compared by code"
            results["info"][oid] = {
                "language": {"declared": man.get("target_lang"), "resolved": dub_lang,
                             "detection": dub_det, "notes": dub_lang_notes},
                "gemini": _gemini_brief(dub_ext),
                "summary": gem_summary,
            }
            _write(run_dir / "measure" / f"{oid}.json", {"drifts": drifts, "speech": speech, "lips_t_len": int(t.size)})
            values_by_out[oid] = values
            sc = scoring.score_output(values, th)
            log.info("%s done: overall=%s coverage=%s%%", oid, sc.get("overall"), sc.get("coverage_pct"))
            results["outputs"][oid] = {"output_id": oid, "vendor": o["vendor"], "file": o["file"],
                                       "values": values, "extras": extras, "score": sc}

        ranks = scoring.rank_outputs({k: v["score"]["overall"] for k, v in results["outputs"].items()})
        for k, r in ranks.items():
            results["outputs"][k]["overall_rank"] = r
        log.info("ranking and pairwise comparison")
        cmp = pairwise.compare_all(values_by_out, th)
        results["pairs"] = cmp["pairs"]
        results["win_matrix"] = cmp["win_matrix"]
        _write(run_dir / "results.json", results)
        _status(run_dir, "done")
        log.info("run finished")
        return results
    except Exception as e:  # keep the error visible in the status file
        log.exception("run failed")
        _status(run_dir, "failed", error=str(e))
        raise


def _speech_total(wav: Path) -> float:
    samples, sr = media.read_wav_mono(wav)
    return round(sum(e - s for s, e in vad.speech_intervals(samples, sr)), 3)


def _gemini_brief(ext: dict) -> dict:
    """Status of one Gemini extraction, without the full segment list (that is in the cache file)."""
    return {"status": ext.get("status"), "reason": ext.get("reason", ""), "model": ext.get("model"),
            "language": ext.get("language"), "segments": len(ext.get("segments", []) or []),
            "dropped_segments": ext.get("dropped_segments", 0)}


def _judge_outputs(run_dir, path, oid, segments, d_w, src_w, values, extras):
    notes, em, tm = [], [], []
    for seg in segments:
        clip = media.trim_clip(path, seg["start"], seg["end"], run_dir / "judge" / f"{seg['seg']:02d}__{oid}.mp4")
        dub_text = " ".join(w["word"] for w in d_w if seg["start"] <= w["start"] < seg["end"])
        measured = {"fidelity": values.get("translation_fidelity"), "onset_ms": values.get("onset_drift_ms")}
        prompt = judge_gemini.build_prompt(seg["text"], dub_text, measured)
        res = judge_gemini.judge_clip(clip, prompt) or {}
        notes.append({"seg": seg["seg"], **res})
        if res.get("status") == "ok":
            em.append(bool(res.get("emotion_match")))
            tm.append(bool(res.get("tone_match")))
    out = {"values": {}, "notes": notes}
    if em:
        out["values"]["emotion_match_pct"] = round(100.0 * sum(em) / len(em), 1)
    if tm:
        out["values"]["tone_match_pct"] = round(100.0 * sum(tm) / len(tm), 1)
    return out
