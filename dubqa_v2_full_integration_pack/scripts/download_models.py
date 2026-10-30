"""One-time model bootstrap. Model cache is configurable and should be a Docker volume."""
import argparse
import os
import urllib.request
from pathlib import Path

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

def download_file(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 100_000:
        print(f"Using cached model: {destination}")
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    if not tmp.exists() or tmp.stat().st_size < 100_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model is missing or unexpectedly small")
    tmp.replace(destination)
    print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")
    return destination

def cache_whisper(model_name: str, cache_dir: Path):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed; install the repository's ML requirements first") from exc
    # Constructing the model downloads/caches it. Use CPU/int8 for a portable bootstrap.
    WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(cache_dir))
    print(f"Cached faster-whisper model: {model_name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-landmarker", action="store_true")
    ap.add_argument("--whisper-model", default=None, help="Optional faster-whisper model, e.g. small")
    ap.add_argument("--cache-dir", default=os.getenv("MODEL_CACHE_DIR", "./model_cache"))
    args = ap.parse_args()
    cache = Path(args.cache_dir).resolve()
    if args.face_landmarker:
        download_file(FACE_MODEL_URL, cache / "mediapipe" / "face_landmarker.task")
    if args.whisper_model:
        cache_whisper(args.whisper_model, cache / "faster-whisper")
    if not args.face_landmarker and not args.whisper_model:
        ap.error("Choose --face-landmarker and/or --whisper-model")

if __name__ == "__main__":
    main()
