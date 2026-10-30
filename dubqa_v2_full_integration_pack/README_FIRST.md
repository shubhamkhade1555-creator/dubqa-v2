# DubQA v2 – Full QA Integration Pack

This is an additive integration pack for the existing public repository:
`https://github.com/shubhamkhade1555-creator/dubqa-v2`

It is **not** a replacement repository and it does not overwrite your existing pipeline. Give
`AGENT_IMPLEMENTATION_PROMPT.md` to your coding agent together with the checked-out repository.
The agent must inspect the current code and adapt the modules to the real call signatures.

## What this pack provides

- A 30-scenario QA catalogue covering lip-sync, word/sentence completeness, translation,
  audio quality, speaker/face association, subtitles, timing and video integrity.
- Excel export with separate sheets for run summary, vendor metrics, word timeline,
  scenario results, defects and run metadata.
- Word report generation **from the Excel workbook**. Excel is the structured source of truth.
- Model download/bootstrap script for the MediaPipe Face Landmarker model and optional
  faster-whisper model cache.
- Docker build/compose guidance with persistent model and run volumes.
- Optional sanitized public-report export. Public publishing is OFF by default.
- Tests for exporters and scenario definitions.

## Important truthfulness rules

- `PASS`, `FAIL`, `REVIEW`, and `NOT_MEASURED` are distinct statuses.
- Missing data is never a pass.
- Current MediaPipe mouth aperture is a visual speech-activity proxy, not phoneme recognition.
- Do not claim phoneme-to-viseme validation unless a real phoneme aligner and a validated
  viseme mapping have been integrated and tested.
- Translation must be evaluated against a target-language reference or semantic review.
  Do not use source-language token overlap as a translation-quality score.
- Competitor output scores must come from actual controlled runs, never marketing claims.

## Quick local setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-ml.txt
pip install -r requirements-qa-report.txt

# Set environment variables using your secret manager or a local .env file.
# Never commit .env.
python scripts/download_models.py --face-landmarker --whisper-model small
python -m pytest -q
```

## Docker

```bash
cp .env.qa.example .env
# Edit .env locally and set GEMINI_API_KEY if needed. Never commit it.
docker compose build
docker compose up
```

The model cache and `runs/` should be mounted as persistent volumes. Model downloads happen
at first bootstrap or via the explicit `download-models` service/command; do not download
large models on every API request.

## Report generation

After a run has produced `runs/<run_id>/results.json`:

```bash
python scripts/export_run_reports.py runs/<run_id>
```

Expected outputs:

- `runs/<run_id>/report/QA_Results.xlsx`
- `runs/<run_id>/report/QA_Report.docx`
- `runs/<run_id>/report/qa_export_manifest.json`

The Word report reads from the Excel workbook, not independently from pipeline JSON.

## Public reports

Default: private/local only.

To create a sanitized public summary deliberately:

```bash
python scripts/export_run_reports.py runs/<run_id> --public-summary
```

The public summary must exclude source/dub media, raw transcripts, names, API keys, local paths,
private notes and personally identifying metadata. Review it before publishing. Never expose
uploaded media automatically just because the GitHub repository is public.

## Integration status

The coding agent must wire the exporter into `server/app.py` after the pipeline and HTML report
finish, add detailed word/phoneme/face evidence where available, and add end-to-end tests.
Do not claim the whole upgrade is integrated until the agent has modified and tested the
actual repository.
