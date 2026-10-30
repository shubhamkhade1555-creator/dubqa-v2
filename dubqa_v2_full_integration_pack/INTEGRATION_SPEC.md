# Integration Specification – DubQA v2

## Existing repository components to preserve

- `core/pipeline.py`: orchestration and results.json creation.
- `core/asr.py`: actual dubbed/source transcript extraction.
- `core/align.py`: current lexical alignment; review its cross-language assumptions.
- `core/lips.py`: MediaPipe mouth landmarks; currently configured for one face in the current
  repository version.
- `core/timing.py`: word onset/tail and A/V timing proxies.
- `core/vad.py`: speech activity.
- `core/info.py`: Gemini segment/speaker context; approximate segment timing must not become
  ground-truth lip-sync timing.
- `core/scoring.py`: current score and coverage logic.
- `report/build_html.py` and `report/templates/report.html`: existing HTML report.
- `server/app.py`: upload/status/results/report endpoints and background worker.
- `Dockerfile`, `docker-compose.yml`, `requirements*.txt`: existing deployment.

## Required integration behavior

1. Keep current API contracts working.
2. Add multi-face tracking as an additive mode. Keep the existing single-face mode available
   until regression tests establish parity.
3. Store face tracks per frame with timestamp, track ID, landmarks/bounding box, visibility
   and confidence. Camera cuts/occlusion must create uncertainty, not forced identity.
4. Store per-word timing: word, start, end, source/dub, speaker ID, face track ID if verified,
   mouth onset, mouth peak, mouth end, onset/tail offset, confidence and status.
5. If a phoneme aligner is not installed/configured, `phoneme_start/end` and
   phoneme-to-viseme status must be `NOT_MEASURED`. Never fabricate phoneme timestamps from
   word timestamps.
6. Compare source meaning to an approved target-language reference and actual dubbed ASR.
   Support a reference transcript file such as `references/<output_id>_expected.txt` or JSON.
   If no target reference exists, semantic completeness is `REVIEW`/`NOT_MEASURED`, not a
   source-vs-target word mismatch FAIL.
7. Detect missing words, extra words, repeated words, missing sentences, truncated starts/ends,
   names/numbers/negations and sentence-order changes. Preserve timestamp evidence.
8. Audio metrics should include duration, integrated loudness (LUFS if tool available),
   true peak, clipping ratio, silence/gap intervals and SNR/noise estimate where reliable.
   Mark unsupported metrics `NOT_MEASURED`.
9. Parse SRT/VTT and compare captions against target-language speech and time intervals.
10. Run the scenario catalogue in `config/qa_scenarios.yaml`; scenarios without valid evidence
    must be `NOT_MEASURED`.
11. Write structured `scenario_results` and `defects` into results.json.
12. Generate Excel from results.json plus detailed word/face/scenario evidence.
13. Generate Word from the Excel workbook only. Store both in the run's `report/` directory.
14. Public summary publishing must be explicit opt-in, sanitized, and never include raw video,
    raw audio, transcripts or API credentials.
15. Keep `results.json` as machine-readable run artifact. Excel is the team-facing source of
    truth; Word is a formatted narrative view generated from Excel.

## Known issues to explicitly inspect and fix

- `core/lips.py` currently uses `num_faces=1` / `max_num_faces=1` in the installed paths.
- `report/build_html.py` currently looks for lip JSON under `precomputed/`, while full pipeline
  writes extracted lip series under `lips/`; make the path resolution consistent.
- `core/pipeline.py` uses lexical alignment between source and dubbed tokens. This is not a
  valid direct translation metric across different languages. Add expected target text and
  semantic review; preserve lexical matching only for same-language comparisons.
- `Dockerfile` currently installs ML requirements unconditionally despite Compose passing a
  `WITH_ML` build argument. Implement and test the build argument.
- Model download/cache paths must be persistent in Docker.
- `.env.example` in the inspected public repository contains what appears to be a credential.
  Replace it with placeholders, rotate the credential, and advise history cleanup. Do not copy
  the secret into any artifact.
- Existing README says faster-whisper word timestamps; inspect the actual `core/asr.py` and
  requirements to reconcile documentation and implementation.
- Current `core/timing.py` mouth aperture logic is a proxy; do not label it phoneme-level
  lip-sync.

## Suggested invocation point

In `server/app.py`, after:

```python
pipeline.run(...)
build_html.build(run_dir)
```

call:

```python
from core.qa_report_bundle import export_run_reports
export_run_reports(run_dir)
```

Use a guarded import or explicit configuration only if necessary, but report generation errors
must be visible in the run log and status. Do not silently swallow them.

## Output layout

```text
runs/<run_id>/
  results.json
  words/
  lips/
  faces/
  measure/
  scenarios/
  report/
    index.html
    QA_Results.xlsx
    QA_Report.docx
    qa_export_manifest.json
    public/
      summary.json          # only when explicitly requested
```
