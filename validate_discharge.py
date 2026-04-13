"""
IFCN Epileptiform Discharge Validator
======================================
Window: 500ms (250ms before + 250ms after annotation)
Criteria: 4/6 IFCN criteria required for positive classification

Author  : Generated for clinical EEG annotation pipeline
Standard: IFCN (International Federation of Clinical Neurophysiology)
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.signal import find_peaks, butter, filtfilt
from dataclasses import dataclass, field
from typing import Optional
import warnings


# ─────────────────────────────────────────────
# Data class for structured output
# ─────────────────────────────────────────────
@dataclass
class DischargeReport:
    is_epileptiform: bool
    criteria_met: int
    discharge_type: str          # 'Spike', 'Sharp Wave', or 'None'

    # Quantitative metrics
    duration_ms: float
    amplitude_ratio: float       # peak vs. background RMS
    slope_ratio: float           # rising slope / falling slope (asymmetry)
    slow_wave_power: float       # relative power in 2-5 Hz post-spike

    # Individual criteria flags
    criteria_details: dict = field(default_factory=dict)

    # Edge-case metadata
    window_truncated: bool = False
    window_samples: int = 0

    def __str__(self) -> str:
        lines = [
            f"{'✓ EPILEPTIFORM' if self.is_epileptiform else '✗ NOT epileptiform'} — {self.discharge_type}",
            f"  Criteria met  : {self.criteria_met}/6",
            f"  Duration      : {self.duration_ms:.1f} ms",
            f"  Amplitude Δ   : {self.amplitude_ratio:.2f}× background RMS",
            f"  Slope ratio   : {self.slope_ratio:.2f} (rise/fall)",
            f"  Slow wave     : {self.slow_wave_power * 100:.1f}% delta power",
            "",
            "  Criteria breakdown:",
        ]
        for key, val in self.criteria_details.items():
            marker = "✓" if val else "✗"
            lines.append(f"    {marker} {key}")
        if self.window_truncated:
            lines.append(f"\n  ⚠ Window truncated — annotation near recording boundary ({self.window_samples} samples used)")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Bandpass filter (0.5–70 Hz, clinical standard)
# ─────────────────────────────────────────────
def _bandpass_filter(segment: np.ndarray, fs: float) -> np.ndarray:
    """Butterworth bandpass filter, order 4. Handles short segments safely."""
    nyq = fs / 2.0
    low = 0.5 / nyq
    high = min(70.0 / nyq, 0.99)   # clamp below Nyquist

    # filtfilt requires at least 3× filter order in samples
    min_len = 27
    if len(segment) < min_len:
        return segment  # too short to filter; return as-is

    b, a = butter(4, [low, high], btype='band')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return filtfilt(b, a, segment)


# ─────────────────────────────────────────────
# Main validator
# ─────────────────────────────────────────────
def validate_epileptiform_discharge(
    signal_data: np.ndarray,
    ann_idx: int,
    fs: float,
    criteria_threshold: int = 4
) -> DischargeReport:
    """
    Validate a potential epileptiform discharge at ann_idx.

    Parameters
    ----------
    signal_data         : 1D numpy array of EEG samples (µV)
    ann_idx             : Sample index of the neurologist's annotation
    fs                  : Sampling frequency in Hz (e.g. 256, 512)
    criteria_threshold  : Minimum IFCN criteria for positive classification (default 4/6)

    Returns
    -------
    DischargeReport with is_epileptiform flag, quantitative metrics,
    and per-criterion flags.
    """
    half_win = int(0.25 * fs)   # 250 ms in samples
    n_total  = len(signal_data)

    # ── EDGE CASE GUARD ────────────────────────────────────────────────
    # Clamp window to valid array bounds; flag if truncated.
    start     = max(0, ann_idx - half_win)
    end       = min(n_total, ann_idx + half_win)
    truncated = (start > ann_idx - half_win) or (end < ann_idx + half_win)

    if end - start < int(0.1 * fs):   # <100ms usable — not enough to analyse
        return DischargeReport(
            is_epileptiform=False, criteria_met=0,
            discharge_type="None", duration_ms=0.0,
            amplitude_ratio=0.0, slope_ratio=0.0,
            slow_wave_power=0.0, window_truncated=True,
            window_samples=end - start,
            criteria_details={"error": "window too short to analyse"}
        )

    window = signal_data[start:end].astype(float)

    # ── DC OFFSET REMOVAL (detrend + mean subtraction) ─────────────────
    # Critical for amplitude accuracy in short (~500 ms) segments.
    window = sp_signal.detrend(window, type='linear')
    window -= window.mean()

    # ── PRE-PROCESSING ──────────────────────────────────────────────────
    filtered = _bandpass_filter(window, fs)

    # Local annotation index within the extracted window
    local_ann = ann_idx - start

    # ═══════════════════════════════════════════════════════════════════
    # CRITERION 1 — Paroxysmal amplitude
    # Peak must stand out vs. background RMS of first+last 100 ms
    # ═══════════════════════════════════════════════════════════════════
    bg_len   = int(0.1 * fs)
    bg_start = filtered[:bg_len]
    bg_end   = (filtered[-bg_len:]
                if len(filtered) >= bg_len * 2
                else filtered[-max(1, len(filtered) // 4):])
    bg_rms   = np.sqrt(np.mean(np.concatenate([bg_start, bg_end]) ** 2))

    peak_amp        = np.abs(filtered[local_ann])
    amplitude_ratio = peak_amp / bg_rms if bg_rms > 1e-9 else 0.0
    crit1_paroxysmal = amplitude_ratio >= 2.5

    # ═══════════════════════════════════════════════════════════════════
    # CRITERIA 2 & 4 — Morphology: slope asymmetry + polarity change
    # ═══════════════════════════════════════════════════════════════════
    derivative  = np.diff(filtered)
    search_half = int(0.05 * fs)    # 50 ms search radius
    d_start     = max(0, local_ann - search_half)
    d_end       = min(len(derivative), local_ann + search_half)

    if d_end - d_start > 2:
        d_region   = derivative[d_start:d_end]
        mid        = len(d_region) // 2
        rise_slope = np.max(np.abs(d_region[:mid])) if mid > 0 else 0.0
        fall_slope = np.max(np.abs(d_region[mid:])) if mid < len(d_region) else 0.0
    else:
        rise_slope = fall_slope = 0.0

    slope_ratio     = rise_slope / fall_slope if fall_slope > 1e-9 else 0.0
    crit2_asymmetry = slope_ratio >= 1.5       # rising ≥ 1.5× faster than falling

    # Polarity change: look for sign flip in derivative near the peak
    near_start = max(0, local_ann - int(0.02 * fs))
    near_end   = min(len(derivative), local_ann + int(0.02 * fs))
    d_near     = derivative[near_start:near_end]
    crit4_polarity = (bool(np.any(np.diff(np.sign(d_near)) != 0))
                      if len(d_near) > 1 else False)

    # ═══════════════════════════════════════════════════════════════════
    # CRITERION 3 — Duration (Spike vs Sharp Wave)
    # Measure trough-to-trough width around the peak.
    # find_peaks on -signal locates local minima (troughs).
    # ═══════════════════════════════════════════════════════════════════
    dur_search = int(0.2 * fs)
    seg_start  = max(0, local_ann - dur_search)
    seg_end    = min(len(filtered), local_ann + dur_search)
    local_seg  = filtered[seg_start:seg_end]
    local_peak = local_ann - seg_start

    inverted       = -local_seg
    troughs, _     = find_peaks(inverted, distance=int(0.01 * fs))

    duration_ms    = 0.0
    discharge_type = "None"
    crit3_duration = False

    if len(troughs) >= 2:
        before = troughs[troughs < local_peak]
        after  = troughs[troughs > local_peak]

        if len(before) > 0 and len(after) > 0:
            t_before    = before[-1]   # closest trough before peak
            t_after     = after[0]    # closest trough after  peak
            duration_ms = ((t_after - t_before) / fs) * 1000.0

            if duration_ms < 70:
                discharge_type = "Spike"
                crit3_duration = True
            elif 70 <= duration_ms <= 200:
                discharge_type = "Sharp Wave"
                crit3_duration = True

    # ═══════════════════════════════════════════════════════════════════
    # CRITERION 5 — Slow wave (2–5 Hz) in post-spike 200 ms
    # ═══════════════════════════════════════════════════════════════════
    post_start = local_ann
    post_end   = min(len(filtered), local_ann + int(0.2 * fs))
    post_seg   = filtered[post_start:post_end]

    slow_wave_power = 0.0
    crit5_slow_wave = False

    if len(post_seg) >= int(0.05 * fs):
        nperseg    = min(len(post_seg), int(0.1 * fs))
        freqs, psd = sp_signal.welch(post_seg, fs=fs, nperseg=nperseg)
        total_power = np.trapezoid(psd, freqs)

        if total_power > 1e-12:
            delta_mask      = (freqs >= 2.0) & (freqs <= 5.0)
            delta_power     = np.trapezoid(psd[delta_mask], freqs[delta_mask])
            slow_wave_power = delta_power / total_power
            crit5_slow_wave = slow_wave_power > 0.30   # >30% power in 2-5 Hz band

    # ═══════════════════════════════════════════════════════════════════
    # CRITERION 6 — Peak sharpness (bonus criterion)
    # High-frequency energy at the peak relative to background
    # ═══════════════════════════════════════════════════════════════════
    sharp_len   = int(0.05 * fs)
    peak_region = filtered[
        max(0, local_ann - sharp_len): min(len(filtered), local_ann + sharp_len)
    ]

    crit6_sharpness = False
    if len(peak_region) > 4:
        peak_energy = np.sum(np.diff(peak_region) ** 2)
        bg_energy   = np.sum(np.diff(filtered) ** 2) / len(filtered)
        crit6_sharpness = (
            (peak_energy / (len(peak_region) * bg_energy)) > 3.0
            if bg_energy > 0 else False
        )

    # ─────────────────────────────────────────────────────────────────
    # Tally criteria and produce final verdict
    # ─────────────────────────────────────────────────────────────────
    criteria_details = {
        "C1_paroxysmal_amplitude": crit1_paroxysmal,
        "C2_slope_asymmetry":      crit2_asymmetry,
        "C3_duration_valid":       crit3_duration,
        "C4_polarity_change":      crit4_polarity,
        "C5_slow_wave_present":    crit5_slow_wave,
        "C6_peak_sharpness":       crit6_sharpness,
    }
    criteria_met    = int(sum(criteria_details.values()))
    is_epileptiform = bool(criteria_met >= criteria_threshold)

    return DischargeReport(
        is_epileptiform=is_epileptiform,
        criteria_met=criteria_met,
        discharge_type=discharge_type,
        duration_ms=round(duration_ms, 2),
        amplitude_ratio=round(amplitude_ratio, 3),
        slope_ratio=round(slope_ratio, 3),
        slow_wave_power=round(slow_wave_power, 4),
        window_truncated=truncated,
        window_samples=end - start,
        criteria_details=criteria_details
    )
