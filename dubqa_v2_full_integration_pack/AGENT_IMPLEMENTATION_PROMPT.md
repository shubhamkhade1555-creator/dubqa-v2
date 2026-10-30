# MASTER CODING-AGENT PROMPT – Integrate Full DubQA v2 QA Upgrade

You are working inside the existing repository:
https://github.com/shubhamkhade1555-creator/dubqa-v2

Your task is to inspect the complete current codebase first, then integrate the QA upgrade
described below into the existing architecture. Do not create a disconnected demo project.
Do not blindly copy files over existing modules. Preserve existing behavior and adapt every
integration to the actual signatures, data structures and tests you find.

## NON-NEGOTIABLE WORK RULES

1. Start by reading the complete repository tree, README, architecture docs, every module in
   `core/`, `server/`, `report/`, `config/`, `scripts/`, Docker files and all tests.
2. Run the existing test suite before changing anything. Record baseline failures.
3. Inspect Git status and current branch. Create a feature branch such as
   `feat/full-qa-evidence-reporting`; do not commit to main.
4. Build a dependency map and integration plan before editing.
5. Keep the current API endpoints and HTML report working.
6. Do not replace the current pipeline with a parallel pipeline. Extend `core/pipeline.py`.
7. Do not invent results, model outputs, confidence values, phoneme timestamps or competitor
   scores. Unknown values must be `NOT_MEASURED`.
8. Do not expose secrets. The checked-in `.env.example` was observed to contain a credential:
   remove it from tracked files, use placeholders, rotate the credential, and provide safe
   Git-history remediation instructions. Never print or copy the credential.
9. Do not upload user media, transcripts, face crops or personal data to public folders.
10. At completion, report exact changed files, tests run, test results, remaining limitations,
    Docker commands and report output paths.

## TARGET CAPABILITIES

### A. Lip-sync and word/phoneme timing

For every dubbed word, store:
- `word`, `word_start`, `word_end`
- `speaker_id`, `face_track_id`
- `mouth_start`, `mouth_peak`, `mouth_end`
- `start_offset_ms`, `end_offset_ms`
- `mouth_tracking_confidence`, `face_visibility_confidence`
- `phonemes[]` with phoneme, start, end and aligner confidence only when a real phoneme
  aligner is available
- `viseme`/viseme interval only when a validated phoneme-to-viseme map is implemented
- `lip_sync_status`: PASS / FAIL / REVIEW / NOT_MEASURED

Important:
- Existing mouth aperture + word timestamp matching is a coarse visual activity proxy.
- It is not phoneme recognition or true phoneme-to-viseme validation.
- Do not label word boundaries as phoneme boundaries.
- Calibrate thresholds by FPS and language. Report frame-level timing resolution.
- Missing face, occlusion, side profile, low confidence and scene cut should cause REVIEW or
  NOT_MEASURED, not automatic FAIL/PASS.
- Use audio diarization plus visual mouth activity for active-speaker evidence. One visible
  face is not proof of speaker identity.

### B. Multi-face tracking and speaker mapping

- Upgrade MediaPipe FaceLandmarker to support configurable `num_faces > 1`.
- Save per-sample frame timestamp, track ID, bbox/landmarks, visibility and confidence.
- Implement track association robustly enough for head motion and short occlusions; use scene
  cuts to reset/reconcile tracks.
- Preserve identity only when tracking evidence supports it.
- Speaker diarization must be separate from face tracking.
- Match diarized speech intervals to mouth-activity intervals using overlap and confidence.
- Handle off-screen speech, reaction shots, simultaneous speakers and camera cuts.
- Store uncertainty. Never use face identity recognition; only associate tracks within the clip.

### C. Completeness and translation tests

For each sentence/word, keep three distinct texts:
1. original source transcript,
2. approved/expected target-language translation,
3. actual target-language transcript from dubbed audio.

Implement:
- missing word / phrase / sentence detection
- extra words and repeated words
- truncated beginning/end
- sentence reorder / merge / split
- critical entities: names, dates, amounts, measurements, technical terms
- negation and polarity changes
- target language mismatch / untranslated fragments
- semantic adequacy and grammar/naturalness review.

Critical rule:
Do not compare source-language tokens directly to target-language tokens and call the
difference missing words. Use an approved target-language reference where possible; otherwise
use semantic evaluation and label uncertainty. Use Gemini as an evaluator with structured
output, but never treat Gemini approximate segment times as timing ground truth.

### D. Audio quality

Add deterministic signal metrics where technically valid:
- audio duration and sample rate
- integrated loudness (LUFS) and true peak (dBTP), if a reliable meter is installed
- clipped sample ratio
- silence/gap intervals
- SNR/noise estimate only when the estimator is valid
- channel imbalance
- abrupt cuts and audio-video duration mismatch.

Also support human listening review for distortion, robotic voice, pronunciation, intelligibility,
echo and source-voice leakage. A loudness metric alone is not speech intelligibility.

### E. Caption QA

Parse SRT and VTT:
- caption text completeness
- target transcript mismatch
- start/end offset relative to speech
- missing/extra caption
- reading speed and overlap
- speaker label and safe-area review where measurable.

### F. Scenario test suite

Use `config/qa_scenarios.yaml` from this pack as the starting catalogue. Implement at least:
- LS: single speaker, late/early mouth, early/late mouth end, long sentence drift, silence,
  multi-face, speaker switch, camera cut, occlusion, off-screen speaker, overlap, fast speech.
- WC: missing one word, multiple words, full sentence, beginning/end truncation, extra word,
  repeated word, wrong replacement, numbers/names/negation.
- TQ: meaning, grammar, terminology, context, tone, untranslated phrase.
- AQ: noise, clipping, low/high loudness, gaps, distortion review, channel imbalance.
- SP: wrong visible speaker, listener reaction, off-screen speaker, multiple simultaneous
  speakers, face identity continuity.
- CC: missing/wrong captions, timing offsets, reading speed, wrong language.
- VI: black/frozen frames, decode errors, FPS/resolution changes, audio/video duration.

Every scenario result must include:
`scenario_id`, `category`, `status`, `severity`, `observed`, `expected`, `evidence_start`,
`evidence_end`, `evidence_refs`, `confidence`, `notes`.

A scenario without enough input/evidence must be NOT_MEASURED. Include injected synthetic
fault tests to prove that detection logic works.

### G. Defect register and scoring

Each defect must include:
`defect_id`, `category`, `severity`, `description`, `expected`, `actual`, `start`, `end`,
`evidence`, `status`, `reviewer_note`.

Use severity:
- CRITICAL: missing full sentence, reversed negation, wrong key speaker, corrupt output.
- HIGH: major lip-sync mismatch, multiple missing words, severe audio issue, wrong key number/name.
- MEDIUM: localized timing drift, unnatural phrase, caption mismatch.
- LOW: minor punctuation/layout issue.

Keep PASS / FAIL / REVIEW / NOT_MEASURED separate.
A critical defect can block release regardless of aggregate score.
Missing metrics reduce coverage; they are never counted as passes.

### H. Excel workbook is the source of truth

Implement an Excel export with `openpyxl`. Create one workbook per run:
`runs/<run_id>/report/QA_Results.xlsx`

Required sheets:
1. `Executive Summary` – run ID, source/target language, mode, overall status, score,
   coverage, vendor count, defect counts and release recommendation.
2. `Vendor Metrics` – one row per output/vendor and all numeric metrics/statuses.
3. `Word Timeline` – one row per word, including word times, phoneme data when present,
   face/speaker IDs, mouth start/peak/end, offsets, confidence and status.
4. `Scenario Results` – every scenario ID, expected/observed, status, severity, timestamps.
5. `Defects` – complete defect register.
6. `Audio Quality` – LUFS, true peak, clipping, SNR, gaps and NOT_MEASURED explanations.
7. `Captions` – cue text, start/end, matched speech interval, status.
8. `Competitor Benchmark` – platform, version/mode, language pair, run ID, metrics, cost,
   processing time and evidence. Keep untested platforms NOT_TESTED; do not infer scores.
9. `Run Metadata` – timestamps, software version, git SHA, input hashes, model names/versions,
   configuration/threshold version.

Add filters, freeze panes, readable widths, conditional formatting and a clear legend.
Use stable column names. Store evidence file paths relative to the run folder.
Never put API keys or raw secrets in the workbook.

### I. Word report generated from Excel

Create:
`runs/<run_id>/report/QA_Report.docx`

The Word report MUST read the generated workbook as its input. Do not independently rebuild
the report from results.json, because Excel is the single team-facing source of truth.

Sections:
1. Cover and run information
2. Executive summary and release recommendation
3. Overall category results
4. Vendor/competitor comparison (only measured rows; N/A and NOT_TESTED remain explicit)
5. Lip-sync findings with word/phoneme evidence
6. Missing words/sentences and translation findings
7. Audio quality findings
8. Speaker mapping and multi-face findings
9. Caption and video integrity findings
10. Critical/high defects with timestamps
11. Scenario pass/fail/review coverage
12. Limitations and NOT_MEASURED items
13. Appendix: run metadata, model versions and thresholds.

Include charts if helpful, but never invent data. Each defect should show a timestamp and
evidence reference when available.

### J. Docker and model downloads

Inspect current Dockerfile and docker-compose.yml before changing them.

Required:
- Correctly implement `WITH_ML` build argument. The current Dockerfile installs ML packages
  unconditionally even though Compose passes the argument.
- Install FFmpeg/ffprobe and required OS libraries.
- Add report dependencies: `openpyxl` and `python-docx`.
- Use a persistent model cache volume (e.g. `/models` or `/root/.cache`).
- Add explicit model bootstrap script and a `download-models` compose profile/service or
  documented one-time command.
- Download MediaPipe Face Landmarker task model into a persistent cache; verify nonzero size
  and ideally checksum.
- Pre-cache the selected faster-whisper model only if the existing ASR implementation actually
  uses faster-whisper. Inspect `core/asr.py` first and reconcile requirements.
- Do not download models on every API request.
- Provide CPU default and optional NVIDIA GPU profile only if CUDA dependencies are pinned
  and tested.
- Keep secrets in environment/secret manager, never in Docker image layers.
- Add health check and persistent `runs/` volume.
- Test both base/precomputed mode and full ML mode.

### K. Public report publishing

The GitHub repository is public. That does NOT mean run artifacts should be public.
Default `REPORT_PUBLISH_MODE=private`.
Only when explicitly set to `sanitized_public` may the app create a public summary.

Sanitized public output may contain:
- run ID alias, date, language pair if approved
- category scores, counts and aggregate metrics
- anonymized defect IDs and broad time windows if approved.

It must not contain:
- uploaded source/dub video or audio
- raw transcripts, captions or face crops
- names, emails, account identifiers
- local filesystem paths
- API keys, environment variables or secrets
- private reviewer comments.

Require a final review/approval flag before publishing. Never automatically commit reports to
the public repository.

## IMPLEMENTATION SEQUENCE

Phase 0 – Audit:
- inspect all files and run tests
- report baseline status
- identify current actual ASR implementation and data schemas.

Phase 1 – Evidence model:
- define typed schemas for word timeline, face track, scenario result, defect and run metadata
- add schema tests and migration-safe defaults.

Phase 2 – Core QA:
- multi-face tracking
- diarization/face association
- completeness and target-reference comparison
- audio quality
- captions
- video integrity
- scenario engine.

Phase 3 – Pipeline:
- wire modules into current `core/pipeline.py`
- preserve existing metrics and results fields
- add `word_timeline`, `scenario_results`, `defects`, `audio_quality`, `captions`, `face_tracks`.

Phase 4 – Reports:
- Excel export
- Word-from-Excel generator
- integrate into `server/app.py` worker after pipeline and HTML report
- expose safe download endpoints for XLSX/DOCX
- optional sanitized public summary only.

Phase 5 – Docker:
- fix build args, model cache, bootstrap, report dependencies and health checks
- document exact commands.

Phase 6 – Validation:
- run all existing tests
- add synthetic known-truth tests for missing word, missing sentence, extra/repeated word,
  word timing offset, mouth onset/tail, false movement, speaker switch, clipping, audio gap,
  caption offset and video corruption.
- test precomputed mode without model downloads
- test full mode with model cache
- test Excel and Word are created and readable
- test Word values are sourced from workbook
- test public summary excludes forbidden fields
- run Docker build and compose smoke test if Docker is available.

## DEFINITION OF DONE

Do not declare completion until:
- Existing tests pass (or baseline failures are documented and not worsened).
- New tests pass.
- All new statuses obey PASS/FAIL/REVIEW/NOT_MEASURED semantics.
- Excel contains every output, metric, scenario, defect and word-level evidence available.
- Word report is generated from Excel and includes the run ID.
- Docker precomputed and full modes are documented and tested.
- Model downloads persist across container restarts.
- Secret leak is removed from tracked files and rotation/history-cleanup instructions are included.
- Public report publishing is disabled by default and has privacy tests.
- The final response lists exact files changed, commands run, results, remaining gaps and artifacts.

Begin with the audit. Do not start by writing code before you understand the existing repository.
