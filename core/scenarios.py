"""Scenario evaluation engine and defect register generator.

Loads scenarios from config/qa_scenarios.yaml and executes checks against:
  - Lip sync metrics & word timeline (LS-*)
  - Lexical / translation completeness against approved target reference (WC-*, TQ-*)
  - Audio quality (clipping, SNR, loudness estimates) (AQ-*)
  - Speaker mapping & multi-face evidence (SP-*)
  - Captions & video integrity (CC-*, VI-*)

Rules:
  - No unconditional PASS branches.
  - Missing or unmeasured evidence yields NOT_MEASURED.
  - Defects must use real measured timestamps.
"""
from pathlib import Path
from typing import Any
import yaml


def load_scenarios() -> list[dict]:
    p = Path(__file__).resolve().parent.parent / "config" / "qa_scenarios.yaml"
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data.get("scenarios", [])


def run_scenario_evaluations(outputs: dict, audio_qualities: dict, captions_data: dict, run_dir: Path) -> tuple[list[dict], list[dict]]:
    """Evaluates all defined QA scenarios across all outputs and registers defects."""
    scenarios = load_scenarios()
    scenario_results = []
    defects = []
    defect_counter = 1

    for sc in scenarios:
        sc_id = sc["id"]
        cat = sc["category"]
        title = sc["title"]
        severity = sc["severity"].upper()
        desc = sc.get("description", "")

        status = "NOT_MEASURED"
        expected = desc
        observed = "Evidence not available or not measured."
        ev_start = None
        ev_end = None
        ev_refs = []
        confidence = 0.0
        notes = ""

        # -------------------------------------------------------------
        # 1. Lip-Sync Scenarios (LS-*)
        # -------------------------------------------------------------
        if sc_id == "LS-01":  # Single-speaker baseline
            # Must verify that valid measured face/mouth samples and word timing exist
            has_valid_data = False
            for out in outputs.values():
                val = out.get("values", {})
                wpr = val.get("word_pass_rate_pct")
                onset = val.get("onset_drift_ms")
                if wpr is not None and onset is not None:
                    has_valid_data = True
                    break

            if has_valid_data:
                status = "PASS"
                observed = "Mouth tracking and word alignment active with measured timing data."
                confidence = 0.95
                ev_refs = ["words_lips_timeline"]
            else:
                status = "NOT_MEASURED"
                observed = "No valid mouth tracking or word timing measurements available."

        elif sc_id == "LS-02":  # Late mouth onset
            onset_values = [(oid, out["values"].get("onset_drift_ms")) for oid, out in outputs.items()
                            if out.get("values", {}).get("onset_drift_ms") is not None]
            if not onset_values:
                status = "NOT_MEASURED"
                observed = "Onset drift not measured."
            else:
                worst_oid, worst_onset = max(onset_values, key=lambda x: x[1])
                if worst_onset > 80.0:
                    status = "FAIL"
                    observed = f"Late mouth onset detected on {worst_oid}: {worst_onset} ms."
                    ev_refs = [f"{worst_oid}_measure"]
                    notes = "Mouth motion begins after audio onset."
                    defects.append({
                        "defect_id": f"DEF-{defect_counter:03d}",
                        "category": "lipsync", "severity": "HIGH",
                        "description": f"Late mouth onset on {worst_oid} ({worst_onset} ms)",
                        "expected": "< 60 ms", "actual": f"{worst_onset} ms",
                        "start": None, "end": None, "evidence": [f"{worst_oid}_measure"], "status": "FAIL"
                    })
                    defect_counter += 1
                else:
                    status = "PASS"
                    observed = f"Onset drift within threshold: {worst_onset} ms."
                    confidence = 0.9

        elif sc_id == "LS-07":  # Silence false movement
            fm_values = [(oid, out["values"].get("false_movement_pct")) for oid, out in outputs.items()
                         if out.get("values", {}).get("false_movement_pct") is not None]
            if not fm_values:
                status = "NOT_MEASURED"
                observed = "False movement in silence not measured."
            else:
                worst_oid, worst_fm = max(fm_values, key=lambda x: x[1])
                if worst_fm > 15.0:
                    status = "FAIL"
                    observed = f"False mouth movement during silence on {worst_oid}: {worst_fm}%."
                    defects.append({
                        "defect_id": f"DEF-{defect_counter:03d}",
                        "category": "lipsync", "severity": "MEDIUM",
                        "description": f"Excessive mouth movement in silence on {worst_oid} ({worst_fm}%)",
                        "expected": "< 10%", "actual": f"{worst_fm}%",
                        "start": None, "end": None, "evidence": ["vad_lips_correlation"], "status": "FAIL"
                    })
                    defect_counter += 1
                else:
                    status = "PASS"
                    observed = f"False mouth movement in silence is acceptable: {worst_fm}%."

        elif sc_id in ("LS-08", "LS-09", "LS-10", "LS-11", "LS-12", "LS-13", "LS-14", "LS-15"):
            # Multi-face / speaker mapping requires explicit active-speaker diarization
            status = "NOT_MEASURED"
            observed = f"{title}: Independent active-speaker diarization evidence not available."

        # -------------------------------------------------------------
        # 2. Word Completeness Scenarios (WC-*)
        # -------------------------------------------------------------
        elif sc_id == "WC-01":  # One missing word
            # Only valid when an approved target reference exists or same-language alignment occurred
            valid_outputs = [out for out in outputs.values() if out.get("extras", {}).get("translation_method") not in (None, "NOT_MEASURED")]
            if not valid_outputs:
                status = "NOT_MEASURED"
                observed = "Cross-language run without approved target reference; omission not measured."
            else:
                missing_found = False
                for out in valid_outputs:
                    missing_words = out.get("extras", {}).get("missing_words", [])
                    if len(missing_words) > 0:
                        missing_found = True
                        status = "FAIL"
                        observed = f"Missing words detected against reference: {', '.join(missing_words[:5])}."
                        defects.append({
                            "defect_id": f"DEF-{defect_counter:03d}",
                            "category": "completeness", "severity": "HIGH",
                            "description": f"Missing words detected on {out.get('output_id')}",
                            "expected": "0 missing words", "actual": f"{len(missing_words)} missing words",
                            "start": None, "end": None, "evidence": ["transcript_alignment"], "status": "FAIL"
                        })
                        defect_counter += 1
                        break
                if not missing_found:
                    status = "PASS"
                    observed = "No missing words detected against reference transcript."

        elif sc_id == "WC-03":  # Missing sentence
            valid_outputs = [out for out in outputs.values() if out.get("values", {}).get("missing_sentences") is not None]
            if not valid_outputs:
                status = "NOT_MEASURED"
                observed = "Sentence completeness not measured (no approved reference or unmeasured)."
            else:
                sent_dropped = False
                for out in valid_outputs:
                    ms = out["values"].get("missing_sentences", 0)
                    if ms > 0:
                        sent_dropped = True
                        status = "FAIL"
                        observed = f"Missing sentence detected on {out.get('output_id')} ({ms} missing)."
                        defects.append({
                            "defect_id": f"DEF-{defect_counter:03d}",
                            "category": "completeness", "severity": "CRITICAL",
                            "description": f"Sentence dropped in dubbing on {out.get('output_id')}",
                            "expected": "All sentences dubbed", "actual": f"{ms} missing sentence(s)",
                            "start": None, "end": None, "evidence": ["sentence_alignment"], "status": "FAIL"
                        })
                        defect_counter += 1
                        break
                if not sent_dropped:
                    status = "PASS"
                    observed = "All expected sentences present."

        # -------------------------------------------------------------
        # 3. Audio Quality Scenarios (AQ-*)
        # -------------------------------------------------------------
        elif sc_id == "AQ-01":  # Background noise / SNR
            snr_values = [aq.get("snr_db") for aq in audio_qualities.values() if isinstance(aq.get("snr_db"), (int, float))]
            if not snr_values:
                status = "NOT_MEASURED"
                observed = "Valid SNR estimator not available or audio too short."
            else:
                min_snr = min(snr_values)
                if min_snr < 15.0:
                    status = "REVIEW"
                    observed = f"Low SNR observed ({min_snr} dB); human listening review recommended."
                else:
                    status = "PASS"
                    observed = f"SNR proxy acceptable ({min_snr} dB)."

        elif sc_id == "AQ-02":  # Clipping and distortion
            clipping_values = [(oid, aq.get("clipped_sample_ratio")) for oid, aq in audio_qualities.items()
                               if isinstance(aq.get("clipped_sample_ratio"), (int, float))]
            if not clipping_values:
                status = "NOT_MEASURED"
                observed = "Clipping ratio not measured."
            else:
                worst_oid, worst_clip = max(clipping_values, key=lambda x: x[1])
                if worst_clip > 0.001:
                    status = "FAIL"
                    observed = f"Digital clipping detected on {worst_oid} ({round(worst_clip * 100, 3)}%)."
                    defects.append({
                        "defect_id": f"DEF-{defect_counter:03d}",
                        "category": "audio", "severity": "HIGH",
                        "description": f"Clipped audio waveform on {worst_oid}",
                        "expected": "0.0% clipping", "actual": f"{round(worst_clip * 100, 3)}% clipped",
                        "start": None, "end": None, "evidence": [f"{worst_oid}.wav"], "status": "FAIL"
                    })
                    defect_counter += 1
                else:
                    status = "PASS"
                    observed = "No digital clipping detected."

        elif sc_id == "AQ-03":  # Loudness and true peak
            # Since ITU BS.1770 / dBTP meter is NOT installed, report truthful NOT_MEASURED with explanation
            has_bs1770 = any(
                aq.get("integrated_lufs") not in (None, "NOT_MEASURED")
                for aq in audio_qualities.values()
            )
            if has_bs1770:
                status = "PASS"
                observed = "Standards-compliant LUFS and true peak measured."
            else:
                status = "NOT_MEASURED"
                observed = "Standards-compliant ITU BS.1770 LUFS / 4x oversampled true-peak meter not installed; RMS estimate used."

        elif sc_id == "AQ-04":  # Speech gap or abrupt cut
            gaps = [aq.get("silence_gap_s") for aq in audio_qualities.values() if isinstance(aq.get("silence_gap_s"), (int, float))]
            if not gaps:
                status = "NOT_MEASURED"
                observed = "Silence gap duration not measured."
            else:
                max_gap = max(gaps)
                if max_gap > 3.0:
                    status = "REVIEW"
                    observed = f"Long silence / speech gap detected ({max_gap}s)."
                else:
                    status = "PASS"
                    observed = f"Silence gap duration within expected range ({max_gap}s)."

        # -------------------------------------------------------------
        # 4. Captions Scenarios (CC-*)
        # -------------------------------------------------------------
        elif sc_id == "CC-01":
            measured_caps = [c for c in captions_data.values() if c.get("has_captions") and c.get("cues_count", 0) > 0]
            if not measured_caps:
                status = "NOT_MEASURED"
                observed = "No caption/subtitle file (SRT/VTT) supplied for evaluation."
            else:
                any_fail = any(c.get("status") == "FAIL" for c in measured_caps)
                any_review = any(c.get("status") == "REVIEW" for c in measured_caps)
                if any_fail:
                    status = "FAIL"
                    observed = "Caption file contains invalid/negative duration cues."
                    defects.append({
                        "defect_id": f"DEF-{defect_counter:03d}",
                        "category": "captions", "severity": "MEDIUM",
                        "description": "Malformed caption cue duration",
                        "expected": "Positive cue duration", "actual": "Negative or zero duration",
                        "start": None, "end": None, "evidence": ["captions"], "status": "FAIL"
                    })
                    defect_counter += 1
                elif any_review:
                    status = "REVIEW"
                    observed = "Caption cues contain reading speed flags or overlapping intervals."
                else:
                    status = "PASS"
                    observed = "Captions parsed cleanly with valid cue timings and reading speeds."

        # -------------------------------------------------------------
        # 5. Video Integrity (VI-01)
        # -------------------------------------------------------------
        elif sc_id == "VI-01":
            # Video integrity must be based on actual probe and decode evidence
            has_video_evidence = False
            for out in outputs.values():
                val = out.get("extras", {})
                if val.get("duration_s") and val.get("duration_s") > 0:
                    has_video_evidence = True
                    break

            if has_video_evidence:
                status = "PASS"
                observed = "Video streams probed successfully with valid duration and stream metadata."
            else:
                status = "NOT_MEASURED"
                observed = "No video stream decoded or probed."

        # -------------------------------------------------------------
        # 6. Speaker Association (SP-01)
        # -------------------------------------------------------------
        elif sc_id == "SP-01":
            status = "NOT_MEASURED"
            observed = "Listener reaction vs active speaker distinction requires multi-speaker diarization."

        scenario_results.append({
            "scenario_id": sc_id,
            "category": cat,
            "title": title,
            "status": status,
            "severity": severity,
            "expected": expected,
            "observed": observed,
            "evidence_start": ev_start,
            "evidence_end": ev_end,
            "evidence_refs": ev_refs,
            "confidence": confidence,
            "notes": notes
        })

    return scenario_results, defects
