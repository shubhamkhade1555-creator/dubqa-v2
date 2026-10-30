"""Lip tracking.

Two sources:
  1. MediaPipe FaceLandmarker (tasks.vision API) on real video.
  2. Precomputed JSON (used by the QC test set and for re-running scoring).

Lip opening = distance(upper inner lip, lower inner lip) / distance(mouth corners).
"""
import json
import urllib.request
from pathlib import Path

import numpy as np

UPPER, LOWER, LEFT, RIGHT = 13, 14, 61, 291
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def opening_from_landmarks(lm) -> float:
    def d(a, b):
        return float(np.hypot(lm[a][0] - lm[b][0], lm[a][1] - lm[b][1]))
    return d(UPPER, LOWER) / max(d(LEFT, RIGHT), 1e-6)


def _ensure_task_model() -> Path:
    p = Path(__file__).resolve().parent.parent / "config" / "models" / "face_landmarker.task"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size < 1000:
        urllib.request.urlretrieve(MODEL_URL, str(p))
    return p


def track_video_mediapipe(video: Path, fps_sample: float = 25.0):
    """Return (t_seconds, opening) arrays using MediaPipe FaceLandmarker."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        raise RuntimeError("MediaPipe/OpenCV not installed; use requirements-ml.txt") from e

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps_sample)))

    # Support both legacy mediapipe solutions and modern mediapipe.tasks.vision
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        ts, vals, idx = [], [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                res = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                t = idx / src_fps
                if res.multi_face_landmarks:
                    lm = [(p.x, p.y) for p in res.multi_face_landmarks[0].landmark]
                    vals.append(opening_from_landmarks(lm))
                else:
                    vals.append(np.nan)
                ts.append(t)
            idx += 1
        cap.release()
        face.close()
        return np.array(ts), np.array(vals, dtype=float)

    # Modern mediapipe Tasks Vision API
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = _ensure_task_model()
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    ts, vals, idx = [], [], 0
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round((idx / src_fps) * 1000))
                res = landmarker.detect_for_video(mp_img, timestamp_ms)
                t = idx / src_fps
                if res.face_landmarks:
                    lm = [(p.x, p.y) for p in res.face_landmarks[0]]
                    vals.append(opening_from_landmarks(lm))
                else:
                    vals.append(np.nan)
                ts.append(t)
            idx += 1
    cap.release()
    return np.array(ts), np.array(vals, dtype=float)


def load_precomputed(path: Path):
    """JSON: {"fps": 25, "opening": [...]} or {"t": [...], "opening": [...]}"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    op = np.array(data["opening"], dtype=float)
    if "t" in data:
        t = np.array(data["t"], dtype=float)
    else:
        t = np.arange(op.size) / float(data.get("fps", 25.0))
    return t, op
