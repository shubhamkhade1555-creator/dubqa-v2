import html
from pathlib import Path

files_to_pack = [
    ('core/asr.py', 'Gemini Audio Speech Recognition', 'Direct audio upload to Gemini 2.5 Flash API with structured JSON output schema extracting word timestamps (start, end) and detected language. Replaces Whisper.'),
    ('core/lips.py', 'MediaPipe Lip Movement Tracking', 'Uses Google MediaPipe Tasks Vision FaceLandmarker on sampled video frames to calculate vertical inner lip opening normalized by mouth corner width.'),
    ('core/pipeline.py', 'Master Orchestration Pipeline', 'Coordinates audio extraction, Gemini ASR, MediaPipe lip tracking, drift math, lexical alignment, LLM emotion evaluation, and saves all JSON artifacts locally.'),
    ('core/align.py', 'Lexical Word Alignment & Matching', 'Fast Dynamic Time Warping (DTW) and edit distance matching between source and dubbed words, detecting missing and hallucinated words.'),
    ('core/timing.py', 'Drift & Synchronization Mathematics', 'Calculates onset drift (ms), tail drift (ms), false movement percentage during silence, open ratio, and audio-visual offsets.'),
    ('core/scoring.py', 'Weighted Multi-Metric Scoring', 'Applies YAML thresholds and metric group weights to compute overall scores (0-100) and coverage percentages without vendor bias.'),
    ('core/pairwise.py', 'Head-to-Head Pairwise Comparison Matrix', 'Compares every dubbed output against every other output across all metrics to compute wins, losses, ties, and tournament rankings.'),
    ('core/judge_gemini.py', 'Gemini Per-Segment Emotion Judge', 'Calls Gemini with video segment payloads to evaluate emotion match, tone preservation, and semantic integrity.'),
    ('core/vad.py', 'Voice Activity Detection (Energy VAD)', 'Computes audio RMS energy envelope to detect voiced speech intervals and inter-word pauses without external AI models.'),
    ('core/media.py', 'FFmpeg Audio & Video Utilities', 'Probes video metadata, extracts 16kHz mono WAV streams, and reads raw PCM buffers for VAD and signal processing.'),
    ('core/lang.py', 'Language Detection & Resolution', 'Combines ASR declared language with unicode script frequency heuristics and Gemini confidence to resolve spoken languages.'),
    ('core/manifest.py', 'Run Manifest Validator', 'Loads and validates run_id, source video specs, and unique output identifiers against schema constraints.'),
    ('server/app.py', 'FastAPI Backend & REST Endpoints', 'Asynchronous web server hosting the dashboard UI, handling multipart video uploads, polling status, and streaming reports.'),
    ('Dockerfile', 'Docker Container Definition', 'Container build file with FFmpeg, Linux OpenGL graphics dependencies (libgl1, libegl1), and Python ML requirements.'),
    ('docker-compose.yml', 'Docker Compose Service Topology', 'Port mappings (8000:8000), volume mounts for runs/, and environment file bindings.'),
    ('requirements.txt', 'Core Python Dependencies', 'FastAPI, Uvicorn, Google GenAI SDK, NumPy, SciPy, PyYAML, Jinja2, Pytest.'),
    ('requirements-ml.txt', 'Machine Learning Dependencies', 'Google MediaPipe, OpenCV-headless for vision processing.'),
    ('config/thresholds.yaml', 'Metric Weights & Threshold Gates', 'Defines Good/Bad pass tolerances, metric group weightings, and release gate rules.'),
    ('config/vendors.yaml', 'Vendor Labels & Display Metadata', 'Labels and platform identifiers for HeyGen, Sync.so, Synthesia, and FrameX.')
]

header = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DubQA v2 — Complete Architectural Handbook & Codebase</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; margin: 0; padding: 30px; }
  h1 { color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
  h2 { color: #7ee787; margin-top: 30px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
  h3 { color: #ffa657; margin-top: 20px; }
  .badge { display: inline-block; background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb; border-radius: 20px; padding: 2px 10px; font-size: 12px; margin-bottom: 10px; }
  .desc { color: #8b949e; font-size: 14px; margin-bottom: 10px; }
  pre { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; overflow-x: auto; font-family: 'SF Mono', Consolas, Monaco, monospace; font-size: 13px; color: #e6edf3; }
  .arch-box { background: #161b22; border: 1px solid #388bfd; border-radius: 10px; padding: 20px; margin-bottom: 30px; }
  ol { padding-left: 20px; }
  li { margin-bottom: 8px; }
</style>
</head>
<body>
<h1>DubQA v2 — Complete Architecture & Codebase Handbook</h1>
<div class="arch-box">
  <h2>System Architecture & End-to-End Pipeline</h2>
  <p>DubQA v2 is an automated audio-visual Quality Assurance system for evaluating dubbed videos against ground-truth source videos.</p>
  <h3>End-to-End Data Flow</h3>
  <ol>
    <li><strong>Input Ingestion:</strong> Source video & 1+ Dubbed video files uploaded via Web UI or REST API.</li>
    <li><strong>Audio Processing:</strong> FFmpeg demuxes 16kHz mono WAV audio tracks; Energy VAD detects speech intervals.</li>
    <li><strong>Gemini ASR Engine:</strong> Direct audio upload to Google Gemini 2.5 Flash API with structured JSON output schema for word timestamps and language ID.</li>
    <li><strong>MediaPipe Vision Engine:</strong> Tasks Vision FaceLandmarker measures upper and lower inner lip distance normalized by mouth width over sampled video frames.</li>
    <li><strong>Drift & Alignment Math:</strong> Dynamic Time Warping aligns words; onset/tail drifts and speech open ratios during silence are measured.</li>
    <li><strong>Gemini LLM Judge (Optional):</strong> Evaluates emotion preservation and tone match per dialogue segment.</li>
    <li><strong>Scoring & Pairwise Matrix:</strong> Weighted scores, Good/Bad gates, and complete head-to-head tournament rankings are computed.</li>
    <li><strong>Artifact Generation:</strong> Local JSON matrices and a standalone interactive HTML report with synchronized video players are generated.</li>
  </ol>
</div>
<h2>Core Source Code Files</h2>
"""

parts = [header]

for rel, title, desc in files_to_pack:
    p = Path(rel)
    if p.exists():
        code = p.read_text(encoding='utf-8')
        parts.append(f"""
<section style="margin-bottom: 40px;">
  <h3>{rel} — {title}</h3>
  <div class="desc">{desc}</div>
  <pre><code>{html.escape(code)}</code></pre>
</section>
""")

parts.append("</body></html>")

out_file = Path("dubqa_v2_complete_handbook.html")
out_file.write_text("".join(parts), encoding="utf-8")
print(f"Generated handbook with {len(files_to_pack)} modules at {out_file.resolve()}")

