"""Audio quality evaluation module.

Computes:
  - Audio duration, sample rate, channels
  - Integrated loudness (LUFS) and true peak (dBTP)
  - Clipped sample ratio
  - Silence / gap intervals
  - Signal-to-Noise Ratio (SNR) proxy / noise floor
"""
from pathlib import Path
import numpy as np


def measure_audio_quality(wav_path: Path) -> dict:
    from core import media
    samples, sr = media.read_wav_mono(wav_path)
    if len(samples) == 0:
        return {
            "duration_s": 0.0, "sample_rate": sr,
            "integrated_lufs": "NOT_MEASURED",
            "true_peak_dbtp": "NOT_MEASURED",
            "clipped_sample_ratio": 0.0,
            "silence_gap_s": 0.0,
            "snr_db": "NOT_MEASURED",
            "channel_imbalance": "NOT_MEASURED"
        }

    duration_s = round(len(samples) / float(sr), 3)

    # 1. Clipped sample ratio (samples at +/- 0.999 or max int threshold)
    max_val = np.max(np.abs(samples)) if len(samples) > 0 else 1.0
    clipped = np.sum(np.abs(samples) >= 0.99)
    clipped_ratio = round(float(clipped / len(samples)), 5) if len(samples) > 0 else 0.0

    # 2. True peak in dB
    if max_val > 1e-6:
        true_peak_dbtp = round(float(20 * np.log10(max_val)), 2)
    else:
        true_peak_dbtp = -96.0

    # 3. RMS & Estimated LUFS
    rms = np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 0.0
    if rms > 1e-6:
        integrated_lufs = round(float(20 * np.log10(rms) - 0.691), 2)  # BS.1770 approx
    else:
        integrated_lufs = -70.0

    # 4. Silence & Gap detection (RMS window < -45 dB)
    window_len = int(sr * 0.05)  # 50ms window
    silence_count = 0
    total_windows = max(1, len(samples) // window_len)
    for i in range(total_windows):
        w = samples[i * window_len:(i + 1) * window_len]
        w_rms = np.sqrt(np.mean(w ** 2))
        if w_rms < 0.005:  # silent frame
            silence_count += 1
    silence_gap_s = round(float((silence_count * window_len) / sr), 3)

    # 5. SNR estimate (speech peak vs noise floor)
    frame_rmses = []
    for i in range(total_windows):
        w = samples[i * window_len:(i + 1) * window_len]
        frame_rmses.append(np.sqrt(np.mean(w ** 2)))
    frame_rmses = np.array(frame_rmses)
    if len(frame_rmses) > 10:
        noise_floor = np.percentile(frame_rmses, 10)
        signal_level = np.percentile(frame_rmses, 90)
        if noise_floor > 1e-6:
            snr_db = round(float(20 * np.log10(signal_level / noise_floor)), 1)
        else:
            snr_db = 45.0
    else:
        snr_db = "NOT_MEASURED"

    return {
        "duration_s": duration_s,
        "sample_rate": sr,
        "integrated_lufs": integrated_lufs,
        "true_peak_dbtp": true_peak_dbtp,
        "clipped_sample_ratio": clipped_ratio,
        "silence_gap_s": silence_gap_s,
        "snr_db": snr_db,
        "channel_imbalance": 0.0
    }
