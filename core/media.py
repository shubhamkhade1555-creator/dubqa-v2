"""FFmpeg / ffprobe helpers. All media work goes through here."""
import json
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{res.stderr[-800:]}")
    return res


def probe(path: Path) -> dict:
    res = _run(["ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path)])
    info = json.loads(res.stdout)
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "duration_s": round(float(info["format"].get("duration", 0.0)), 3),
        "has_video": v is not None,
        "has_audio": a is not None,
        "width": v.get("width") if v else None,
        "height": v.get("height") if v else None,
        "fps": v.get("r_frame_rate") if v else None,
    }


def extract_audio_wav(src: Path, dst: Path, sr: int = 16000) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr),
          "-acodec", "pcm_s16le", str(dst)])
    return dst


def read_wav_mono(path: Path):
    """Return (samples float32 in [-1,1], sample_rate) using the stdlib wave module."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
        width = w.getsampwidth()
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        arr = (arr - 128.0) / 128.0
    else:
        arr = arr / float(np.iinfo(dtype).max)
    return arr, sr


def trim_clip(src: Path, start_s: float, end_s: float, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(src),
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-c:a", "aac", "-b:a", "64k",
          "-movflags", "+faststart", str(dst)])
    return dst


def make_web_copy(src: Path, dst: Path, height: int = 360) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(src), "-vf", f"scale=-2:{height}",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "32", "-c:a", "aac", "-b:a", "48k",
          "-ac", "1", "-movflags", "+faststart", str(dst)])
    return dst
