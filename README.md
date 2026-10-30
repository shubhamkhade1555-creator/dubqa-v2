# DubQA v2

Measures dubbed videos against the source video and against each other.

- **Local measurement** (no AI): lip movement (MediaPipe Face Mesh), speech activity (energy VAD), word timing (Whisper word timestamps), lexical alignment, A/V offset, false mouth movement during silence.
- **Gemini** (optional): judges only emotion, tone and meaning per segment. It never measures timing or lip sync.
- **Outputs**: a scored scorecard per dub, Good/Bad labels per metric against thresholds, a composite score with coverage, competition ranking with ties, a pairwise head-to-head matrix (every dub vs every other dub), and a self-contained HTML report with synchronised players and lip timelines.

Measured values are never adjusted to favour or penalise any vendor. Missing values are shown as "Not measured" and reduce coverage; they are never counted as a pass.

## Layout

```
config/thresholds.yaml   metric thresholds, group weights, release gate, tie tolerances
config/vendors.yaml      vendor labels (display only)
core/                    manifest, media (ffmpeg), vad, lips, timing, align, asr, scoring, pairwise, judge_gemini, pipeline
report/                  build_html.py and templates/report.html
server/app.py            FastAPI: upload a run, poll status, fetch results and report
scripts/make_synthetic.py  builds the QC test batch with known timing offsets
tests/                   unit and end-to-end tests
runs/<run_id>/           one folder per run (inputs, measure, results.json, report/)
```

## Run a run

Each run is a folder under `runs/` with:

```
runs/<run_id>/
  manifest.json
  inputs/  src__en__clip01.mp4, <output_id>__<lang>__clip01.mp4 ...
  precomputed/   (optional) <output_id>_words.json, <output_id>_lips.json
```

`manifest.json` lists the source file, the target language, and each dubbed output's `output_id` (must match `^[a-z0-9_\-]+$`, unique).

Modes:
- `precomputed`: uses words/lips JSON from `precomputed/`. No ML needed. Used by the QC batch.
- `full`: runs Whisper (faster-whisper) and MediaPipe on the videos. Requires `requirements-ml.txt`.

Gemini judging runs only when `use_judge` is set and `GEMINI_API_KEY` is present. Without a key it is skipped and reported as not measured.

## Languages

- Source and target language default to `auto`. Detection order: Whisper's detected language (full mode), a `language` field in the words JSON, Gemini's language, then the Unicode script of the transcript.
- A language typed in the form wins. A disagreement with the detected language is recorded in the report, not overwritten.
- Latin-script languages cannot be separated by script alone, so set them in the form if Whisper and Gemini are not available.

## Gemini extraction (language, speakers, emotions, counts)

With **Use Gemini** ticked and `GEMINI_API_KEY` set, each video is sent to Gemini once. Gemini returns language, speakers, and segments (approximate start and end, speaker, emotion, tone, transcript). Code then computes:

- word count and syllable count per segment and in total (syllables are a heuristic),
- segment count and speaker count, source against dub,
- emotion agreement for segments that pair up by time overlap.

Gemini's times are never used for lip-sync or timing numbers. The required-data checklist in the report shows what was present and what was missing.

See `docs/ARCHITECTURE.md` for the diagram and module map.

## Setup (local)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml.txt      # only for mode=full
cp .env.example .env                     # then set GEMINI_API_KEY
python -m pytest -q                      # unit, language/count and fake-Gemini end-to-end tests
python scripts/make_synthetic.py         # rebuild the QC batch
```

Requires `ffmpeg` and `ffprobe` on the PATH.

## Setup (Docker)

```bash
cp .env.example .env                     # set GEMINI_API_KEY (never commit .env)
docker compose build                     # add --build-arg WITH_ML=1 for full mode
docker compose up
```

The app listens on `http://localhost:8000`. Runs are written to `./runs` on the host.

### Using the web app

1. Open `http://localhost:8000`.
2. Enter a run name, the source and target language codes, and the measurement mode.
3. Choose the source video. In precomputed mode, also choose its words and lips JSON.
4. Add one row per dubbed video. Pick its platform from the dropdown. The platform is pre-selected when the filename contains the platform name, for example `heygen_hindi.mp4`.
5. Click **Start run**. The page shows the status, the live log, a ranked summary, and the report when the run finishes.

The log is `runs/<run_id>/run.log`. Every line is also written to the server console.

There is no login. Run the app only on a trusted network, or put it behind your own access control before exposing it.

Endpoints:
- `POST /api/runs` multipart: `manifest` (JSON), `files` (videos), `mode`, `use_judge`
- `GET /api/runs/{run_id}` status (`running` / `done` / `failed`)
- `GET /api/runs/{run_id}/results` JSON results
- `GET /api/runs/{run_id}/report` HTML report

## QC test batch

`scripts/make_synthetic.py` writes a run with four videos (source plus HeyGen, Sync.so, FrameX style dubs) and precomputed timing JSON. The designed offsets are:

| Output  | Onset (designed) | Tail (designed) | Shift | Notes |
|---------|------------------|-----------------|-------|-------|
| HeyGen  | 100 ms           | 30 ms           | 150 ms | |
| Sync.so | 60 ms            | 60 ms           | 50 ms  | blink in mid-silence |
| FrameX  | 20 ms            | -20 ms          | 20 ms  | drops word 3 (w2) |

Expected measured values are within about ±40 ms (one frame at 25 fps). The ranks in this batch are an artifact of the synthetic design and say nothing about the real vendors.

Known limitation: a blink that lands right after a word boundary is counted in the tail measurement. Keep test blinks in mid-silence.

## Known gaps

- Translation fidelity uses a lexical proxy (matched words / source words). COMET is not wired.
- A/V offset is a coarse cross-correlation estimate.
- The FastAPI server and Docker image were not executed in the build sandbox. Run `docker compose up` and check `/docs` before relying on them.
- The Gemini judge has not been run against a live key.
- Syllable counting is a heuristic: vowel-nucleus rules for Devanagari, vowel groups for Latin script.
- Whitespace word counting does not split CJK text.
- No XLSX export yet.

## Security

Do not commit `.env`. If a Gemini key has been pasted into a chat or shared file, rotate it and set the new key through the environment.
