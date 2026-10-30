"""Lip and face tracking.

Supports:
  1. Multi-face tracking with bounding-box derivation, landmark mouth aperture,
     gated centroid association, and track persistence (track_faces_mediapipe).
  2. Single-face backward-compatible mouth aperture (track_video_mediapipe).
  3. Precomputed JSON loading for tests and offline evaluation.
"""
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

UPPER, LOWER, LEFT, RIGHT = 13, 14, 61, 291
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def opening_from_landmarks(lm) -> float:
    def d(a, b):
        return float(np.hypot(lm[a][0] - lm[b][0], lm[a][1] - lm[b][1]))
    return d(UPPER, LOWER) / max(d(LEFT, RIGHT), 1e-6)


def _ensure_task_model() -> Path:
    cache_dir = os.environ.get("MODEL_CACHE_DIR")
    if cache_dir:
        cand = Path(cache_dir) / "mediapipe" / "face_landmarker.task"
        if cand.exists() and cand.stat().st_size > 100_000:
            return cand

    p = Path(__file__).resolve().parent.parent / "config" / "models" / "face_landmarker.task"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size < 1000:
        urllib.request.urlretrieve(MODEL_URL, str(p))
    return p


class FaceTracker:
    """Stable multi-face tracker with gated distance matching and track aging."""

    def __init__(self, max_distance: float = 0.25, max_age_frames: int = 25):
        self.max_distance = max_distance
        self.max_age_frames = max_age_frames
        self.tracks: dict[str, dict[str, Any]] = {}  # track_id -> {cx, cy, last_seen_frame, bbox}
        self.next_id = 1

    def update(self, detected_faces: list[dict], frame_idx: int) -> list[dict]:
        """detected_faces: list of dicts with 'cx', 'cy', 'bbox', 'op', etc.
        Returns list of matched face records for the current frame.
        """
        # Age out old tracks that haven't been matched
        stale_ids = [
            tid for tid, tdata in self.tracks.items()
            if (frame_idx - tdata["last_seen_frame"]) > self.max_age_frames
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        matched_records = []
        used_track_ids = set()

        for d_face in detected_faces:
            best_track_id = None
            min_dist = self.max_distance

            for tid, tdata in self.tracks.items():
                if tid in used_track_ids:
                    continue
                dist = float(np.hypot(d_face["cx"] - tdata["cx"], d_face["cy"] - tdata["cy"]))
                if dist < min_dist:
                    min_dist = dist
                    best_track_id = tid

            if best_track_id is None:
                best_track_id = f"face_{self.next_id}"
                self.next_id += 1

            used_track_ids.add(best_track_id)
            self.tracks[best_track_id] = {
                "cx": d_face["cx"],
                "cy": d_face["cy"],
                "bbox": d_face["bbox"],
                "last_seen_frame": frame_idx
            }

            matched_records.append({
                "track_id": best_track_id,
                "bbox": d_face["bbox"],
                "mouth_opening": d_face["op"],
                "confidence": d_face.get("confidence", "NOT_MEASURED"),
                "visibility": d_face.get("visibility", "NOT_MEASURED")
            })

        return matched_records


def track_faces_mediapipe(video: Path, num_faces: int = 3, fps_sample: float = 25.0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Tracks multiple faces and returns (ts, primary_opening, face_tracks)."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        raise RuntimeError("MediaPipe/OpenCV not installed; use requirements-ml.txt") from e

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps_sample)))

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = _ensure_task_model()
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=max(1, num_faces),
    )

    ts, primary_vals, idx = [], [], 0
    face_tracks_records = []
    tracker = FaceTracker(max_distance=0.25, max_age_frames=int(fps_sample * 2))

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = idx / src_fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(t * 1000))
                res = landmarker.detect_for_video(mp_img, timestamp_ms)

                current_frame_tracks = []
                if res.face_landmarks:
                    detected_faces = []
                    for f_landmarks in res.face_landmarks:
                        lm = [(p.x, p.y) for p in f_landmarks]
                        xs = [p[0] for p in lm]
                        ys = [p[1] for p in lm]
                        min_x, max_x = max(0.0, float(min(xs))), min(1.0, float(max(xs)))
                        min_y, max_y = max(0.0, float(min(ys))), min(1.0, float(max(ys)))
                        bbox = [round(min_x, 4), round(min_y, 4), round(max_x - min_x, 4), round(max_y - min_y, 4)]
                        cx = float(np.mean(xs))
                        cy = float(np.mean(ys))
                        op = opening_from_landmarks(lm)
                        detected_faces.append({
                            "lm": lm, "cx": cx, "cy": cy, "bbox": bbox, "op": round(op, 4),
                            "confidence": "NOT_MEASURED",
                            "visibility": "visible"
                        })

                    frame_assigned = tracker.update(detected_faces, frame_idx=idx)
                    for rec in frame_assigned:
                        item = {
                            "t": round(t, 3),
                            "track_id": rec["track_id"],
                            "bbox": rec["bbox"],
                            "mouth_opening": rec["mouth_opening"],
                            "confidence": rec["confidence"],
                            "visibility": rec["visibility"]
                        }
                        face_tracks_records.append(item)
                        current_frame_tracks.append(item)

                    primary_face = next((f for f in current_frame_tracks if f["track_id"] == "face_1"), current_frame_tracks[0])
                    primary_vals.append(primary_face["mouth_opening"])
                else:
                    primary_vals.append(np.nan)

                ts.append(t)
            idx += 1

    cap.release()
    return np.array(ts), np.array(primary_vals, dtype=float), face_tracks_records


def track_video_mediapipe(video: Path, fps_sample: float = 25.0):
    """Backward compatible single-face tracker."""
    ts, primary_vals, _ = track_faces_mediapipe(video, num_faces=1, fps_sample=fps_sample)
    return ts, primary_vals


def load_precomputed(path: Path):
    """JSON: {"fps": 25, "opening": [...]} or {"t": [...], "opening": [...]}"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    op = np.array(data["opening"], dtype=float)
    if "t" in data:
        t = np.array(data["t"], dtype=float)
    else:
        t = np.arange(op.size) / float(data.get("fps", 25.0))
    return t, op
