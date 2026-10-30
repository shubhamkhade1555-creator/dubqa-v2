"""Audio quality evaluation module.

Computes:
  - Audio duration, sample rate, channels
  - RMS loudness estimate (dBFS) and sample peak (dBFS) [truthful labeling]
  - Clipped sample ratio
  - Silence / gap intervals
  - Signal-to-Noise Ratio (SNR) proxy / noise floor
  - Multichannel channel imbalance
"""
from pathlib import Path
import numpy as np


def measure_audio_quality(wav_path: Path, original_multichannel_path: Path | None = None) -> dict:
    from core import media
    samples, sr = media.read_wav_mono(wav_path)
    if len(samples) == 0:
        return {
            "duration_s": 0.0, "sample_rate": sr,
            "rms_loudness_estimate_dbfs": "NOT_MEASURED",
            "sample_peak_dbfs": "NOT_MEASURED",
            "integrated_lufs": "NOT_MEASURED",
            "true_peak_dbtp": "NOT_MEASURED",
            "clipped_sample_ratio": 0.0,
            "silence_gap_s": 0.0,
            "snr_db": "NOT_MEASURED",
            "channel_imbalance": "NOT_MEASURED"
        }

    duration_s = round(len(samples) / float(sr), 3)

    # 1. Clipped sample ratio (samples at +/- 0.99 or max threshold)
    max_val = np.max(np.abs(samples)) if len(samples) > 0 else 1.0
    clipped = np.sum(np.abs(samples) >= 0.99)
    clipped_ratio = round(float(clipped / len(samples)), 5) if len(samples) > 0 else 0.0

    # 2. Sample peak in dBFS
    if max_val > 1e-6:
        sample_peak_dbfs = round(float(20 * np.log10(max_val)), 2)
    else:
        sample_peak_dbfs = -96.0

    # 3. RMS Loudness Estimate in dBFS (truthfully labeled)
    rms = np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 0.0
    if rms > 1e-6:
        rms_loudness_estimate_dbfs = round(float(20 * np.log10(rms)), 2)
    else:
        rms_loudness_estimate_dbfs = -70.0

    # 4. Silence & Gap detection (RMS window < -45 dB)
    window_len = max(1, int(sr * 0.05))  # 50ms window
    silence_count = 0
    total_windows = max(1, len(samples) // window_len)
    for i in range(total_windows):
        w = samples[i * window_len:(i + 1) * window_len]
        w_rms = np.sqrt(np.mean(w ** 2))
        if w_rms < 0.005:  # silent frame (~ -46 dBFS)
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
            snr_db = round(float(20 * np.log10(max(signal_level, 1e-6) / noise_floor)), 1)
        else:
            snr_db = 45.0
    else:
        snr_db = "NOT_MEASURED"

    # 6. Channel imbalance from multichannel audio if available
    channel_imbalance = "NOT_MEASURED"
    if original_multichannel_path and Path(original_multichannel_path).exists():
        try:
            import soundfile as sf
            data, _ = sf.read(str(original_multichannel_path))
            if data.ndim > 1 and data.shape[1] > 1:
                left_rms = np.sqrt(np.mean(data[:, 0] ** 2))
                right_rms = np.sqrt(np.mean(data[:, 1] ** 2))
                if left_rms > 1e-6 and right_rms > 1e-6:
                    channel_imbalance = round(float(abs(20 * np.log10(left_rms / right_rms))), 2)
        except Exception:
            channel_imbalance = "NOT_MEASURED"

    return {
        "duration_s": duration_s,
        "sample_rate": sr,
        "rms_loudness_estimate_dbfs": rms_loudness_estimate_dbfs,
        "sample_peak_dbfs": sample_peak_dbfs,
        "integrated_lufs": "NOT_MEASURED",  # Standards-compliant ITU BS.1770 requires specialized filterbank
        "true_peak_dbtp": "NOT_MEASURED",   # True-peak requires 4x oversampling interpolation
        "clipped_sample_ratio": clipped_ratio,
        "silence_gap_s": silence_gap_s,
        "snr_db": snr_db,
        "channel_imbalance": channel_imbalance
    }
