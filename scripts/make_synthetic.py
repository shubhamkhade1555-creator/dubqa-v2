"""Build a QC test batch with KNOWN timing errors, so the pipeline can be checked against truth.

Usage: python scripts/make_synthetic.py runs/QC_synthetic

Designed truth (per output, relative to each dub word's own start):
  heygen   : mouth opens +100 ms, closes +30 ms after word end, timing shift +150 ms
  sync     : mouth opens +60 ms, closes +60 ms, shift +50 ms, blinks in silence (false movement)
  framex   : mouth opens +20 ms, closes -20 ms (early), shift +20 ms, drops word 3 (missing word)
Expected: onset medians 100 / 60 / 20 ms, tail medians 30 / 60 / 20 ms, FrameX missing 1 word.
"""
import json
import math
import sys
import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 16000
DUR = 12.0
FPS = 25
WORDS = [f"w{i}" for i in range(6)]
WORD_START = [1.0 + i * 1.8 for i in range(6)]
WORD_LEN = 0.8

SOURCE_ID = "src__en__clip01"
OUTPUTS = [
    {"output_id": "heygen__ms__clip01", "vendor": "HeyGen", "shift": 0.15, "onset": 0.10, "tail": 0.03,
     "drop": [], "blink": False},
    {"output_id": "sync__ms__clip01", "vendor": "Sync.so", "shift": 0.05, "onset": 0.06, "tail": 0.06,
     "drop": [], "blink": True},
    {"output_id": "framex__ms__clip01", "vendor": "FrameX", "shift": 0.02, "onset": 0.02, "tail": -0.02,
     "drop": [2], "blink": False},
]


def write_wav(path: Path, speech: list[tuple[float, float]]) -> None:
    t = np.arange(int(DUR * RATE)) / RATE
    x = np.zeros_like(t)
    for s, e in speech:
        m = (t >= s) & (t < e)
        x[m] = 0.5 * np.sin(2 * math.pi * 220 * t[m])
    pcm = (x * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm)


def mux(wav: Path, mp4: Path) -> None:
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate={FPS}:duration={DUR}",
           "-i", str(wav), "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "35",
           "-c:a", "aac", "-b:a", "64k", str(mp4)]
    subprocess.run(cmd, check=True, capture_output=True)


def sentences(words: list[dict]) -> list[dict]:
    out = []
    for k in range(0, len(words), 2):
        pair = words[k:k + 2]
        out.append({"text": " ".join(w["word"] for w in pair),
                    "start": pair[0]["start"], "end": pair[-1]["end"]})
    return out


def lips_series(open_windows: list[tuple[float, float]], blinks: list[tuple[float, float]]) -> list[float]:
    t = np.arange(0, DUR, 1.0 / FPS)
    op = np.full_like(t, 0.02)
    for s, e in open_windows + blinks:
        op[(t >= s) & (t < e)] = 0.5
    return op.round(4).tolist()


def build(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "precomputed").mkdir(parents=True, exist_ok=True)
    src_words = [{"word": WORDS[i], "start": WORD_START[i], "end": WORD_START[i] + WORD_LEN}
                 for i in range(6)]
    src_speech = [(w["start"], w["end"]) for w in src_words]
    write_wav(run_dir / "work_src.wav", src_speech)
    mux(run_dir / "work_src.wav", run_dir / "inputs" / f"{SOURCE_ID}.mp4")
    (run_dir / "precomputed" / "source_words.json").write_text(
        json.dumps({"words": src_words, "sentences": sentences(src_words)}, indent=2))
    (run_dir / "precomputed" / "source_lips.json").write_text(json.dumps(
        {"fps": FPS, "opening": lips_series([(w["start"], w["end"]) for w in src_words], [])}))

    manifest_outputs = []
    for o in OUTPUTS:
        oid = o["output_id"]
        kept = [i for i in range(6) if i not in o["drop"]]
        d_words = []
        speech = []
        open_w = []
        for i in kept:
            s = WORD_START[i] + o["shift"]
            e = s + WORD_LEN
            d_words.append({"word": WORDS[i], "start": round(s, 3), "end": round(e, 3)})
            speech.append((s, e))
            open_w.append((s + o["onset"], e + o["tail"]))
        blinks = []
        if o["blink"]:
            for i in range(6):
                bs = WORD_START[i] + WORD_LEN + 0.50   # mid-silence, away from word boundaries
                blinks.append((bs, bs + 0.2))
        wav = run_dir / f"work_{oid}.wav"
        write_wav(wav, speech)
        mux(wav, run_dir / "inputs" / f"{oid}.mp4")
        (run_dir / "precomputed" / f"{oid}_words.json").write_text(
            json.dumps({"words": d_words, "sentences": sentences(d_words)}, indent=2))
        (run_dir / "precomputed" / f"{oid}_lips.json").write_text(json.dumps(
            {"fps": FPS, "opening": lips_series(open_w, blinks)}))
        manifest_outputs.append({"output_id": oid, "vendor": o["vendor"], "file": f"{oid}.mp4"})
        wav.unlink()
    for w in run_dir.glob("work_*.wav"):
        w.unlink()
    manifest = {"run_id": run_dir.name, "source": {"file": f"{SOURCE_ID}.mp4", "lang": "en"},
                "target_lang": "ms", "outputs": manifest_outputs}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("synthetic batch written to", run_dir)


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/QC_synthetic"))
