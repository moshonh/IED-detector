"""
IFCN Epileptiform Discharge Validator + Streamlit App
======================================================
References:
  [1] Nascimento & Beniczky, Neurology Education 2023 — 6 criteria definition
  [2] Kural et al., Neurology 2020 — clinical validation, IRA weights, thresholds

IFCN Criteria with IRA weights (Kural 2020, Table 2):
  C1 — Di/tri-phasic sharp morphology (20–200 ms)       IRA AC1 = 0.75  HIGH
  C2 — Duration different from background               IRA AC1 = 0.63  HIGH
  C3 — Waveform asymmetry (rise vs fall)                IRA AC1 = 0.30  LOW
  C4 — After-going slow wave (2–5 Hz)                   IRA AC1 = 0.65  HIGH
  C5 — Disruption of background activity                IRA AC1 = 0.36  LOW
  C6 — Physiologic field/distribution (multi-ch only)   IRA AC1 = 0.51  MODERATE

Scoring modes (Kural 2020, Table 1):
  Clinical  ≥4/5 evaluable → sensitivity 96%, specificity 85%, accuracy 91%
  Strict    ≥5/5 evaluable → sensitivity 81%, specificity 96%, accuracy 88%
  (C6 counted only when multi-channel data is available)

Weighted score (IRA-based, for display only):
  High-IRA criteria (C1, C2, C4) weighted 1.0
  Moderate-IRA (C6)              weighted 0.68
  Low-IRA (C3, C5)               weighted 0.40

Run:    streamlit run validate_discharge.py
Import: from validate_discharge import validate_epileptiform_discharge
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.signal import find_peaks, butter, filtfilt
from dataclasses import dataclass, field
import warnings


# ─────────────────────────────────────────────────────────────────────
# IRA-based weights per criterion (Kural 2020, Table 2)
# ─────────────────────────────────────────────────────────────────────
CRITERION_IRA = {
    "C1": 0.75,   # HIGH  — spiky morphology
    "C2": 0.63,   # HIGH  — duration differs from BG
    "C3": 0.30,   # LOW   — asymmetry
    "C4": 0.65,   # HIGH  — slow after-wave
    "C5": 0.36,   # LOW   — BG disruption
    "C6": 0.51,   # MOD   — field distribution
}

IRA_LABEL = {
    "C1": "HIGH",
    "C2": "HIGH",
    "C3": "LOW ⚠",
    "C4": "HIGH",
    "C5": "LOW ⚠",
    "C6": "MOD",
}


# ─────────────────────────────────────────────────────────────────────
# Data class for structured output
# ─────────────────────────────────────────────────────────────────────
@dataclass
class DischargeReport:
    # Primary verdict
    is_epileptiform_clinical: bool   # ≥4/5 evaluable (sensitivity mode)
    is_epileptiform_strict:   bool   # ≥5/5 evaluable (specificity >95% mode)
    criteria_met: int                # count of True criteria (excl. N/A)
    weighted_score: float            # IRA-weighted sum (0–max ~4.3)
    discharge_type: str              # 'Spike', 'Sharp Wave', 'None'
    phases_detected: int             # C1: number of phases (1, 2, or 3)

    # Per-criterion quantitative values
    duration_ms:          float   # C1
    bg_duration_ms:       float   # C2 reference
    slope_ratio:          float   # C3
    slow_wave_power:      float   # C4
    bg_suppression_ratio: float   # C5
    field_available:      bool    # C6

    # C6 detailed result dict (None if single-channel)
    c6_field_result: dict = field(default_factory=dict)

    # Detailed flags (bool or "n/a")
    criteria_details: dict = field(default_factory=dict)

    # Edge-case metadata
    window_truncated: bool = False
    window_samples:   int  = 0

    def __str__(self) -> str:
        lines = [
            "═" * 56,
            f"  CLINICAL  (≥4/5): {'✓ EPILEPTIFORM' if self.is_epileptiform_clinical else '✗ not epileptiform'}"
            f" — {self.discharge_type}",
            f"  STRICT    (≥5/5): {'✓ EPILEPTIFORM' if self.is_epileptiform_strict   else '✗ not epileptiform'}",
            f"  Criteria met     : {self.criteria_met}/5 evaluable",
            f"  Weighted score   : {self.weighted_score:.2f}  (IRA-adjusted, max ≈4.3)",
            "─" * 56,
            f"  C1 Morphology    : {self.duration_ms:.1f} ms  phases={self.phases_detected}  [IRA HIGH]",
            f"  C2 Duration vs BG: discharge {self.duration_ms/2:.1f} ms  BG {self.bg_duration_ms:.1f} ms  [IRA HIGH]",
            f"  C3 Slope ratio   : {self.slope_ratio:.2f}  (rise/fall)  [IRA LOW ⚠]",
            f"  C4 Slow wave     : {self.slow_wave_power*100:.1f}%  delta power  [IRA HIGH]",
            f"  C5 BG disruption : {self.bg_suppression_ratio:.2f}×  post/pre RMS  [IRA LOW ⚠]",
            f"  C6 Field         : {'N/A (single channel)' if not self.field_available else 'evaluated'}  [IRA MOD]",
            "─" * 56,
        ]
        for k, v in self.criteria_details.items():
            m = "✓" if v is True else ("➖" if v == "n/a" else "✗")
            lines.append(f"    {m}  {k}")
        if self.window_truncated:
            lines.append(f"\n  ⚠ Window truncated ({self.window_samples} samples)")
        lines.append("═" * 56)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Signal processing helpers
# ─────────────────────────────────────────────────────────────────────
def _bandpass_filter(segment: np.ndarray, fs: float) -> np.ndarray:
    """Butterworth bandpass 0.5–70 Hz, order 4."""
    nyq  = fs / 2.0
    low  = 0.5 / nyq
    high = min(70.0 / nyq, 0.99)
    if len(segment) < 27:
        return segment
    b, a = butter(4, [low, high], btype='band')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return filtfilt(b, a, segment)


def _median_half_wave_ms(seg: np.ndarray, fs: float) -> float:
    """Median half-wave duration via zero-crossing intervals."""
    if len(seg) < 4:
        return 0.0
    c = seg - seg.mean()
    s = np.sign(c)
    s[s == 0] = 1
    crossings = np.where(np.diff(s))[0]
    if len(crossings) < 2:
        return 0.0
    return float(np.median(np.diff(crossings) / fs * 1000.0))


def _count_phases(seg: np.ndarray, fs: float, peak_idx: int) -> int:
    """
    Count di- or tri-phasic components around the main peak.
    Looks for zero-crossings within ±100ms of peak.
    Returns 1 (monophasic), 2 (diphasic), or 3 (triphasic).
    """
    radius = min(int(0.1 * fs), peak_idx, len(seg) - peak_idx - 1)
    if radius < 2:
        return 1
    region = seg[peak_idx - radius: peak_idx + radius]
    c = region - region.mean()
    s = np.sign(c)
    s[s == 0] = 1
    crossings = np.where(np.diff(s))[0]
    # Number of phases = number of zero-crossing intervals + 1, capped at 3
    phases = min(len(crossings) + 1, 3)
    return max(1, phases)


# ─────────────────────────────────────────────────────────────────────
# Main validator
# ─────────────────────────────────────────────────────────────────────
def validate_epileptiform_discharge(
    signal_data:          np.ndarray,
    ann_idx:              int,
    fs:                   float,
    channel_count:        int        = 1,
    eeg_matrix:           np.ndarray = None,
    ch_names:             list       = None,
    peak_sample_override: int        = None,
) -> DischargeReport:
    """
    Validate a potential epileptiform discharge using 6 IFCN criteria.

    Parameters
    ----------
    signal_data          : 1D numpy array (µV), single EEG channel
    ann_idx              : Sample index of the neurologist's annotation
    fs                   : Sampling frequency in Hz (256, 512, etc.)
    channel_count        : Number of EEG channels (C6 requires >1)
    eeg_matrix           : (n_ch × n_samples) full multi-channel EEG for C6
    ch_names             : list of channel names matching rows of eeg_matrix
    peak_sample_override : precise peak sample for C6 voltage map

    Returns
    -------
    DischargeReport — contains both clinical (≥4) and strict (≥5) verdicts,
    per-criterion flags, IRA-weighted score, and quantitative metrics.
    """
    half_win  = int(0.25 * fs)
    n_total   = len(signal_data)

    # ── Edge case guard ────────────────────────────────────────────
    start     = max(0, ann_idx - half_win)
    end       = min(n_total, ann_idx + half_win)
    truncated = (start > ann_idx - half_win) or (end < ann_idx + half_win)

    _empty = DischargeReport(
        is_epileptiform_clinical=False, is_epileptiform_strict=False,
        criteria_met=0, weighted_score=0.0, discharge_type="None",
        phases_detected=1, duration_ms=0.0, bg_duration_ms=0.0,
        slope_ratio=0.0, slow_wave_power=0.0, bg_suppression_ratio=0.0,
        field_available=False, window_truncated=True,
        window_samples=end - start,
        criteria_details={"error": "window too short to analyse"}
    )
    if end - start < int(0.1 * fs):
        return _empty

    window    = signal_data[start:end].astype(float)
    window    = sp_signal.detrend(window, type='linear')
    window   -= window.mean()
    filtered  = _bandpass_filter(window, fs)
    local_ann = ann_idx - start

    # ══════════════════════════════════════════════════════════════════
    # C1 — Di/tri-phasic sharp morphology, duration 20–200 ms
    # IRA = 0.75 (HIGH)
    # Kural 2020: "di- or tri-phasic waves with sharp or spiky morphology
    # (i.e., pointed peak)"
    # ══════════════════════════════════════════════════════════════════
    dur_search = int(0.2 * fs)
    seg_s      = max(0, local_ann - dur_search)
    seg_e      = min(len(filtered), local_ann + dur_search)
    local_seg  = filtered[seg_s:seg_e]
    local_peak = local_ann - seg_s

    troughs, _ = find_peaks(-local_seg, distance=max(1, int(0.01 * fs)))

    duration_ms      = 0.0
    discharge_type   = "None"
    phases_detected  = 1
    crit1_morphology = False

    if len(troughs) >= 2:
        before = troughs[troughs < local_peak]
        after  = troughs[troughs > local_peak]
        if len(before) > 0 and len(after) > 0:
            t_before    = before[-1]
            t_after     = after[0]
            duration_ms = ((t_after - t_before) / fs) * 1000.0

            if 20 <= duration_ms < 70:
                discharge_type = "Spike"
            elif 70 <= duration_ms <= 200:
                discharge_type = "Sharp Wave"

            if discharge_type != "None":
                # Count phases (di- / tri-phasic check)
                phases_detected  = _count_phases(filtered, fs, local_ann)
                # C1 passes if duration valid AND at least diphasic
                crit1_morphology = phases_detected >= 2

    # ══════════════════════════════════════════════════════════════════
    # C2 — Duration different from background (either shorter OR longer)
    # IRA = 0.63 (HIGH)
    # Kural 2020: "either shorter or longer" than background
    # ══════════════════════════════════════════════════════════════════
    bg_len        = int(0.1 * fs)
    bg_signal     = np.concatenate([filtered[:bg_len], filtered[-bg_len:]])
    bg_duration_ms = _median_half_wave_ms(bg_signal, fs)
    crit2_duration = False

    if bg_duration_ms > 0 and duration_ms > 0:
        discharge_half = duration_ms / 2.0
        ratio = discharge_half / bg_duration_ms if bg_duration_ms > 1e-3 else 0.0
        # Discharge half-wave differs from BG half-wave (shorter OR longer)
        crit2_duration = bool((ratio < 0.70) or (ratio > 1.50))

    # ══════════════════════════════════════════════════════════════════
    # C3 — Waveform asymmetry
    # IRA = 0.30 (LOW) — Kural 2020: "fair agreement"
    # "sharply rising ascending phase and more slowly decaying descending
    # phase, or vice versa" — either direction counts
    # ══════════════════════════════════════════════════════════════════
    derivative  = np.diff(filtered)
    search_half = int(0.05 * fs)
    d_s         = max(0, local_ann - search_half)
    d_e         = min(len(derivative), local_ann + search_half)
    slope_ratio = 0.0
    crit3_asymmetry = False

    if d_e - d_s > 2:
        d_region   = derivative[d_s:d_e]
        mid        = len(d_region) // 2
        rise_slope = float(np.max(np.abs(d_region[:mid]))) if mid > 0 else 0.0
        fall_slope = float(np.max(np.abs(d_region[mid:]))) if mid < len(d_region) else 0.0
        if fall_slope > 1e-9:
            slope_ratio = rise_slope / fall_slope
        elif rise_slope > 1e-9:
            slope_ratio = 3.0   # only rise, extreme asymmetry
        # Either direction (rise faster or fall faster → ratio far from 1.0)
        crit3_asymmetry = bool(slope_ratio >= 1.5 or slope_ratio <= 0.67)

    # ══════════════════════════════════════════════════════════════════
    # C4 — After-going slow wave (2–5 Hz dominant in post-discharge 200ms)
    # IRA = 0.65 (HIGH)
    # ══════════════════════════════════════════════════════════════════
    post_s    = local_ann
    post_e    = min(len(filtered), local_ann + int(0.2 * fs))
    post_seg  = filtered[post_s:post_e]

    slow_wave_power = 0.0
    crit4_slow_wave = False

    if len(post_seg) >= int(0.05 * fs):
        nperseg = len(post_seg)
        nfft    = max(nperseg, int(fs))   # zero-pad → 1 Hz resolution
        freqs, psd = sp_signal.welch(post_seg, fs=fs, nperseg=nperseg, nfft=nfft)
        total_power = np.trapezoid(psd, freqs)
        if total_power > 1e-12:
            delta_mask      = (freqs >= 2.0) & (freqs <= 5.0)
            delta_power     = np.trapezoid(psd[delta_mask], freqs[delta_mask])
            slow_wave_power = float(delta_power / total_power)
            crit4_slow_wave = bool(slow_wave_power > 0.30)

    # ══════════════════════════════════════════════════════════════════
    # C5 — Disruption of background activity (suppression/flattening)
    # IRA = 0.36 (LOW) — Kural 2020: "fair agreement"
    # Post-discharge RMS < 60% of pre-discharge RMS
    # ══════════════════════════════════════════════════════════════════
    pre_len  = int(0.15 * fs)
    post_len = int(0.15 * fs)
    pre_seg2  = filtered[max(0, local_ann - pre_len): local_ann]
    post_seg2 = filtered[local_ann: min(len(filtered), local_ann + post_len)]

    bg_suppression_ratio = 0.0
    crit5_suppression    = False

    if len(pre_seg2) > 4 and len(post_seg2) > 4:
        pre_rms  = float(np.sqrt(np.mean(pre_seg2 ** 2)))
        post_rms = float(np.sqrt(np.mean(post_seg2 ** 2)))
        if pre_rms > 1e-9:
            bg_suppression_ratio = post_rms / pre_rms
            crit5_suppression    = bool(bg_suppression_ratio < 0.60)

    # ══════════════════════════════════════════════════════════════════
    # C6 — Physiologic field / distribution
    # IRA = 0.51 (MODERATE) — requires multi-channel
    # ══════════════════════════════════════════════════════════════════
    field_available = channel_count > 1
    crit6_field     = "n/a"
    c6_field_result = None

    if field_available and eeg_matrix is not None and ch_names is not None:
        try:
            from topomap import analyse_field
            # Use precise peak sample if provided (auto-channel selection)
            _peak_for_c6 = peak_sample_override if peak_sample_override is not None else ann_idx
            c6_field_result = analyse_field(eeg_matrix, ch_names, _peak_for_c6, fs)
            crit6_field     = bool(c6_field_result["passed"])
        except Exception:
            crit6_field     = "n/a"

    # ─────────────────────────────────────────────────────────────────
    # Tally — binary count (excl. N/A) + IRA-weighted score
    # ─────────────────────────────────────────────────────────────────
    criteria_details = {
        "C1_diphasic_sharp_morphology": bool(crit1_morphology),
        "C2_duration_differs_from_bg":  bool(crit2_duration),
        "C3_waveform_asymmetry":        bool(crit3_asymmetry),
        "C4_aftergoing_slow_wave":      bool(crit4_slow_wave),
        "C5_background_disruption":     bool(crit5_suppression),
        "C6_physiologic_field":         crit6_field,
    }

    ira_map = {
        "C1_diphasic_sharp_morphology": CRITERION_IRA["C1"],
        "C2_duration_differs_from_bg":  CRITERION_IRA["C2"],
        "C3_waveform_asymmetry":        CRITERION_IRA["C3"],
        "C4_aftergoing_slow_wave":      CRITERION_IRA["C4"],
        "C5_background_disruption":     CRITERION_IRA["C5"],
        "C6_physiologic_field":         CRITERION_IRA["C6"],
    }

    bool_criteria = {k: v for k, v in criteria_details.items() if v != "n/a"}
    criteria_met  = int(sum(bool(v) for v in bool_criteria.values()))

    weighted_score = float(sum(
        ira_map[k] for k, v in criteria_details.items()
        if v is True
    ))

    # Kural 2020 thresholds:
    #   ≥4/5 evaluable → clinical use (accuracy 91%, specificity 85%)
    #   ≥5/5 evaluable → strict use   (accuracy 88%, specificity 96%)
    is_epileptiform_clinical = bool(criteria_met >= 4)
    is_epileptiform_strict   = bool(criteria_met >= 5)

    return DischargeReport(
        is_epileptiform_clinical=is_epileptiform_clinical,
        is_epileptiform_strict=is_epileptiform_strict,
        criteria_met=criteria_met,
        weighted_score=round(weighted_score, 3),
        discharge_type=discharge_type,
        phases_detected=phases_detected,
        duration_ms=round(duration_ms, 2),
        bg_duration_ms=round(bg_duration_ms, 2),
        slope_ratio=round(slope_ratio, 3),
        slow_wave_power=round(slow_wave_power, 4),
        bg_suppression_ratio=round(bg_suppression_ratio, 3),
        field_available=field_available,
        c6_field_result=c6_field_result or {},
        window_truncated=truncated,
        window_samples=end - start,
        criteria_details=criteria_details,
    )


# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════
try:
    import streamlit as st
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.signal import detrend

    st.set_page_config(
        page_title="IFCN EEG Validator",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 IFCN Epileptiform Discharge Validator")
    st.caption(
        "Based on: Nascimento & Beniczky 2023 · Kural et al. 2020 · "
        "6 IFCN criteria · IRA-weighted scoring · 500 ms window"
    )

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Signal Settings")
        fs_manual = st.selectbox("Sampling frequency — synthetic only (Hz)",
                                  [256, 512, 1024], index=0,
                                  help="EDF files use their own fs automatically.")

        st.divider()
        st.subheader("Synthetic EEG")
        noise_std       = st.slider("Background noise (µV)",    1.0, 20.0, 5.0, 0.5)
        spike_amp       = st.slider("Spike amplitude (µV)",     0.0, 150.0, 80.0, 5.0)
        spike_width     = st.slider("Spike half-width (ms)",    5.0, 100.0, 20.0, 1.0)
        add_triphasic   = st.checkbox("Add tri-phasic pre-deflection (C1)", value=True)
        sw_amp          = st.slider("Slow wave amplitude (µV)", 0.0, 80.0, 40.0, 5.0)
        sw_freq         = st.slider("Slow wave freq (Hz)",      1.0, 8.0, 3.0, 0.5)
        add_suppression = st.checkbox("Add post-discharge suppression (C5)", value=True)
        seed            = st.number_input("Random seed", 0, 9999, 42, 1)

        st.divider()
        st.subheader("📊 Clinical thresholds (Kural 2020)")
        st.markdown("""
| Mode | Cutoff | Sensitivity | Specificity |
|------|--------|------------|------------|
| Clinical | ≥4/5 | 96% | 85% |
| Strict   | ≥5/5 | 81% | **96%** |
        """)
        st.caption("C6 evaluated automatically for multi-channel EDF/CSV.")

    # ── Signal source ─────────────────────────────────────────────────
    @st.cache_data
    def make_synthetic_app(fs, spike_amp, sw_amp, sw_freq,
                           add_suppression, seed):
        from topomap import make_synthetic_multichannel
        eeg_mat, chs, ann = make_synthetic_multichannel(
            fs=fs, duration=10.0, focus_ch="C3",
            spike_amp=spike_amp, seed=seed
        )
        focus_i = chs.index("C3")
        if sw_amp > 0:
            sw_len = int(0.2 * fs)
            sw_t   = np.linspace(0, 0.2, sw_len)
            if ann + sw_len <= eeg_mat.shape[1]:
                eeg_mat[focus_i, ann:ann+sw_len] += sw_amp * np.sin(2*np.pi*sw_freq*sw_t)
        if add_suppression:
            supp_len = int(0.15 * fs)
            if ann + supp_len <= eeg_mat.shape[1]:
                eeg_mat[focus_i, ann:ann+supp_len] *= 0.25
        return eeg_mat, chs, ann

    # ─────────────────────────────────────────────────────────────────
    # Helper: auto-select dominant channel (peak-to-peak in window)
    # ─────────────────────────────────────────────────────────────────
    def _auto_select_channel(eeg_mat: np.ndarray, ch_names: list,
                              ann_idx: int, fs: float) -> str:
        """
        Return the channel name with the highest peak-to-peak amplitude
        in the 500ms window around ann_idx (Beniczky criterion).
        """
        half   = int(0.25 * fs)
        start  = max(0, ann_idx - half)
        end    = min(eeg_mat.shape[1], ann_idx + half)
        window = eeg_mat[:, start:end]
        ptp    = np.ptp(window, axis=1)   # peak-to-peak per channel
        return ch_names[int(np.argmax(ptp))]

    # ─────────────────────────────────────────────────────────────────
    # Helper: find precise peak sample for voltage map (C6)
    # ─────────────────────────────────────────────────────────────────
    def _find_global_peak_sample(eeg_mat: np.ndarray, focus_ch: str,
                                  ch_names: list, ann_idx: int,
                                  fs: float) -> int:
        """
        Find the sample with maximum absolute amplitude in the focus
        channel within ±60ms of ann_idx.
        Returns the global sample index.
        """
        radius = int(0.06 * fs)
        start  = max(0, ann_idx - radius)
        end    = min(eeg_mat.shape[1], ann_idx + radius)
        ch_idx = ch_names.index(focus_ch)
        seg    = eeg_mat[ch_idx, start:end]
        return start + int(np.argmax(np.abs(seg)))

    # ─────────────────────────────────────────────────────────────────
    # Helper: parse external TXT annotation file (Natus + generic)
    # ─────────────────────────────────────────────────────────────────
    def _parse_txt_annotations(txt_content: str) -> list:
        """
        Parse Natus EEG annotation TXT files and generic formats.

        Natus format (tab-separated, header lines before the data table):
            Time            Duration    Title
            d1 00:16:34                 Beginning of Recording
            d1 00:16:36                 x
            d1 00:16:38                 End of Study

        The parser:
          1. Skips header lines until it finds the column-header row
             (contains 'Time' and 'Title').
          2. Reads 'Beginning of Recording' to establish recording_start_sec.
          3. Converts every subsequent wall-clock timestamp to offset_sec
             relative to recording start.
          4. Skips 'End of Study', 'Beginning of Recording', and empty titles.
          5. Falls back to generic one-timestamp-per-line parsing if the
             Natus header row is never found.

        Returns list of (onset_sec, duration_sec, description).
        """
        import re

        def _parse_wall_clock(s: str):
            """
            Parse 'dN HH:MM:SS[.mmm]' (Natus) or plain 'HH:MM:SS' / 'MM:SS'.
            Returns total seconds as float, or None on failure.
            """
            s = s.strip()
            # Strip 'd1 ', 'd2 ' day prefix used by Natus
            s = re.sub(r"^d\d+\s+", "", s)
            hms = re.match(r"^(\d+):(\d+):(\d+\.?\d*)$", s)
            ms  = re.match(r"^(\d+):(\d+\.?\d*)$", s)
            sec = re.match(r"^(\d+\.?\d*)$", s)
            if hms:
                h, m, sv = hms.groups()
                return int(h)*3600 + int(m)*60 + float(sv)
            if ms:
                m, sv = ms.groups()
                return int(m)*60 + float(sv)
            if sec:
                return float(sec.group(1))
            return None

        lines = txt_content.strip().splitlines()

        # ── Try Natus table format ────────────────────────────────────
        SKIP_TITLES = {"beginning of recording", "end of study",
                       "end of recording", "start of recording"}

        header_idx  = None
        time_col    = None
        dur_col     = None
        title_col   = None

        for i, line in enumerate(lines):
            # Look for the header row containing 'Time' and 'Title'
            parts = [p.strip() for p in line.split("\t")]
            lower = [p.lower() for p in parts]
            if "time" in lower and "title" in lower:
                header_idx = i
                time_col   = lower.index("time")
                title_col  = lower.index("title")
                dur_col    = lower.index("duration") if "duration" in lower else None
                break

        if header_idx is not None:
            recording_start_sec = None
            results = []

            for line in lines[header_idx + 1:]:
                if not line.strip():
                    continue
                cols = [c.strip() for c in line.split("\t")]
                # Pad columns if row is short
                while len(cols) <= max(time_col, title_col):
                    cols.append("")

                time_str  = cols[time_col]  if time_col  < len(cols) else ""
                title_str = cols[title_col] if title_col < len(cols) else ""
                dur_str   = cols[dur_col]   if (dur_col is not None
                                                 and dur_col < len(cols)) else "0"

                wall_sec = _parse_wall_clock(time_str)
                if wall_sec is None:
                    continue

                # Establish recording start from first 'Beginning of Recording'
                if "beginning of recording" in title_str.lower():
                    recording_start_sec = wall_sec
                    continue

                # Skip end markers and blank titles
                if title_str.lower() in SKIP_TITLES or not title_str:
                    continue

                # Compute offset from recording start
                if recording_start_sec is not None:
                    onset_sec = wall_sec - recording_start_sec
                    if onset_sec < 0:
                        onset_sec += 86400  # handle midnight rollover
                else:
                    onset_sec = wall_sec  # no start found, use raw time

                try:
                    duration = float(dur_str)
                except (ValueError, TypeError):
                    duration = 0.0

                results.append((onset_sec, duration, title_str))

            return results

        # ── Generic fallback: one timestamp per line ──────────────────
        results = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in re.split(r"[\t,;]+|  +", line)
                     if p.strip()]
            if not parts:
                continue
            onset_sec = _parse_wall_clock(parts[0])
            if onset_sec is None:
                continue
            try:
                duration = float(parts[1]) if len(parts) > 1 else 0.0
            except (ValueError, TypeError):
                duration = 0.0
            label = parts[2] if len(parts) > 2 else "IED"
            results.append((onset_sec, duration, label))
        return results

    # ─────────────────────────────────────────────────────────────────
    # EDF loader
    # ─────────────────────────────────────────────────────────────────
    @st.cache_data
    def _load_edf(file_bytes: bytes):
        """Load EDF/EDF+. Returns (eeg_matrix, ch_names, fs, annotations)."""
        import pyedflib, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            f        = pyedflib.EdfReader(tmp_path)
            n_ch     = f.signals_in_file
            ch_names = [f.getLabel(i).strip() for i in range(n_ch)]
            fs_out   = float(f.getSampleFrequency(0))
            n_samp   = f.getNSamples()[0]
            eeg_mat  = np.zeros((n_ch, n_samp), dtype=float)
            for i in range(n_ch):
                eeg_mat[i, :] = f.readSignal(i)
            anns = []
            try:
                for onset, dur, desc in zip(*f.readAnnotations()):
                    anns.append((float(onset), float(dur), str(desc).strip()))
            except Exception:
                pass
            f.close()
        finally:
            os.unlink(tmp_path)
        return eeg_mat, ch_names, fs_out, anns

    # ─────────────────────────────────────────────────────────────────
    # UI tabs
    # ─────────────────────────────────────────────────────────────────
    tab_synth, tab_edf = st.tabs(["🧪 Synthetic (demo)", "📂 Upload EDF / EDF+"])

    eeg = eeg_matrix = ch_names = ann_idx = None
    fs  = fs_manual
    source_label     = "Synthetic EEG"
    annotations_list = []
    peak_sample      = None   # precise peak sample for voltage map

    with tab_synth:
        eeg_matrix, ch_names, ann_idx = make_synthetic_app(
            fs, spike_amp, sw_amp, sw_freq, add_suppression, int(seed)
        )
        eeg          = eeg_matrix[ch_names.index("C3")]
        source_label = "Synthetic 19-ch (focus: C3)"
        peak_sample  = ann_idx

    with tab_edf:
        uploaded_edf = st.file_uploader(
            "Upload EDF or EDF+ file. "
            "Annotations are read automatically from the file.",
            type=["edf", "EDF"],
            key="edf_uploader",
        )

        # ── Optional: external TXT annotation file ────────────────────
        with st.expander("📎 Load external annotations (TXT)", expanded=False):
            uploaded_txt = st.file_uploader(
                "TXT file exported from clinical software "
                "(one onset per line: seconds, MM:SS, or HH:MM:SS).",
                type=["txt"],
                key="txt_uploader",
            )
            if uploaded_txt is not None:
                txt_content = uploaded_txt.read().decode("utf-8", errors="replace")
                parsed_txt  = _parse_txt_annotations(txt_content)
                if parsed_txt:
                    st.success(f"Parsed {len(parsed_txt)} annotation(s) from TXT.")
                    st.dataframe(
                        {"Onset (s)": [a[0] for a in parsed_txt],
                         "Duration": [a[1] for a in parsed_txt],
                         "Label":    [a[2] for a in parsed_txt]},
                        use_container_width=True,
                    )
                else:
                    st.warning("No valid timestamps found in TXT file.")
                    parsed_txt = []
            else:
                parsed_txt = []

        if uploaded_edf is not None:
            try:
                eeg_mat_up, ch_names_up, fs_up, anns_edf = _load_edf(
                    uploaded_edf.read()
                )
                from topomap import ELECTRODE_POS_2D
                valid_chs = [c for c in ch_names_up if c in ELECTRODE_POS_2D]
                if not valid_chs:
                    st.warning(
                        f"No standard 10-20 channel names matched. "
                        f"Channels: {ch_names_up[:8]}. C6 will be skipped."
                    )
                    valid_chs = ch_names_up

                n_samp_up = eeg_mat_up.shape[1]

                # ── File info ─────────────────────────────────────────
                ci1, ci2, ci3, ci4 = st.columns(4)
                ci1.metric("Channels",  len(ch_names_up))
                ci2.metric("Fs (Hz)",   f"{fs_up:.0f}")
                ci3.metric("Duration",  f"{n_samp_up/fs_up:.1f} s")
                ci4.metric("EDF anns",  len(anns_edf))

                # ── Merge EDF + TXT annotations ───────────────────────
                all_anns = list(anns_edf) + [
                    (o, d, f"[TXT] {lbl}") for o, d, lbl in parsed_txt
                ]
                all_anns.sort(key=lambda x: x[0])

                # ── Annotation selector ───────────────────────────────
                if all_anns:
                    src_tag = "EDF+TXT" if parsed_txt else "EDF"
                    st.success(
                        f"✅ {len(all_anns)} annotation(s) available [{src_tag}]"
                    )
                    ann_options = [
                        f"[{i+1}]  {o:.3f}s  —  {d or '(no label)'}"
                        for i, (o, dur, d) in enumerate(all_anns)
                    ] + ["── Manual position ──"]

                    chosen = st.selectbox(
                        "Select annotation (neurologist marking):",
                        ann_options,
                    )
                    if chosen == "── Manual position ──":
                        ann_sec    = st.slider(
                            "Manual annotation (seconds)",
                            0.5, n_samp_up/fs_up - 0.5,
                            n_samp_up/fs_up/2, 0.01,
                        )
                        ann_idx_up = int(ann_sec * fs_up)
                        ann_label  = f"manual @ {ann_sec:.3f}s"
                    else:
                        sel        = ann_options.index(chosen)
                        ann_idx_up = int(all_anns[sel][0] * fs_up)
                        ann_label  = (f"{all_anns[sel][2] or 'IED'} "
                                      f"@ {all_anns[sel][0]:.3f}s")
                else:
                    st.info("No annotations found. Use manual slider.")
                    ann_sec    = st.slider(
                        "Annotation position (seconds)",
                        0.5, n_samp_up/fs_up - 0.5,
                        n_samp_up/fs_up/2, 0.01,
                    )
                    ann_idx_up = int(ann_sec * fs_up)
                    ann_label  = f"manual @ {ann_sec:.3f}s"

                # Clamp to valid range
                half_w     = int(0.25 * fs_up)
                ann_idx_up = int(np.clip(
                    ann_idx_up, half_w, n_samp_up - half_w - 1
                ))

                # ── Auto-select dominant channel ──────────────────────
                auto_ch    = _auto_select_channel(
                    eeg_mat_up, ch_names_up, ann_idx_up, fs_up
                )
                auto_valid = auto_ch if auto_ch in valid_chs else (
                    valid_chs[0] if valid_chs else ch_names_up[0]
                )
                st.info(
                    f"🤖 **Auto-selected channel:** `{auto_valid}` "
                    f"(highest peak-to-peak in 500ms window)"
                )
                override = st.checkbox("Override auto-selection", value=False)
                if override:
                    focus_up = st.selectbox(
                        "Focus channel:", valid_chs,
                        index=valid_chs.index(auto_valid)
                              if auto_valid in valid_chs else 0,
                    )
                else:
                    focus_up = auto_valid

                # ── Find precise peak sample for voltage map ──────────
                peak_sample_up = _find_global_peak_sample(
                    eeg_mat_up, focus_up, ch_names_up, ann_idx_up, fs_up
                )
                peak_offset_ms = (peak_sample_up - ann_idx_up) / fs_up * 1000
                st.caption(
                    f"Peak sample: {peak_sample_up}  "
                    f"({peak_offset_ms:+.1f} ms relative to annotation)"
                )

                # ── Commit to session state ───────────────────────────
                eeg_matrix   = eeg_mat_up
                ch_names     = ch_names_up
                ann_idx      = ann_idx_up
                fs           = fs_up
                eeg          = eeg_matrix[ch_names.index(focus_up)]
                peak_sample  = peak_sample_up
                source_label = (f"{uploaded_edf.name} · {focus_up} · "
                                f"{ann_label}")
                annotations_list = all_anns

            except ImportError:
                st.error(
                    "pyedflib not installed. "
                    "Run: pip install pyedflib  then restart."
                )
            except Exception as ex:
                st.error(f"Could not load EDF: {ex}")
                with st.expander("Debug info"):
                    import traceback
                    st.code(traceback.format_exc())

    if eeg is None:   # fallback to synthetic
        eeg_matrix, ch_names, ann_idx = make_synthetic_app(
            fs, spike_amp, sw_amp, sw_freq, add_suppression, int(seed)
        )
        eeg         = eeg_matrix[ch_names.index("C3")]
        peak_sample = ann_idx

    n_ch = eeg_matrix.shape[0] if eeg_matrix is not None else 1

    # ── Run validator (uses precise peak_sample for C6 voltage map) ───
    report = validate_epileptiform_discharge(
        eeg, ann_idx, fs,
        channel_count=n_ch,
        eeg_matrix=eeg_matrix,
        ch_names=ch_names,
        peak_sample_override=peak_sample,
    )

    # ── Dual verdict banner ───────────────────────────────────────────
    col_c, col_s = st.columns(2)
    with col_c:
        if report.is_epileptiform_clinical:
            st.success(
                f"✅ **CLINICAL** (≥4/5): EPILEPTIFORM — {report.discharge_type}\n\n"
                f"Sensitivity 96% · Specificity 85%", icon="⚡"
            )
        else:
            st.warning(
                f"⬜ **CLINICAL** (≥4/5): NOT epileptiform\n\n"
                f"({report.criteria_met}/5 criteria met)", icon="🔍"
            )
    with col_s:
        if report.is_epileptiform_strict:
            st.success(
                f"✅ **STRICT** (≥5/5): EPILEPTIFORM — {report.discharge_type}\n\n"
                f"Sensitivity 81% · Specificity 96%", icon="🎯"
            )
        else:
            st.info(
                f"⬜ **STRICT** (≥5/5): NOT epileptiform\n\n"
                f"({report.criteria_met}/5 criteria met)", icon="🎯"
            )

    if report.window_truncated:
        st.warning(f"⚠️ Window truncated near recording boundary ({report.window_samples} samples used)")

    st.divider()

    # ── Metrics row ───────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    _total_eval = 6 if report.field_available else 5
    m1.metric("Criteria met",     f"{report.criteria_met}/{_total_eval}",
              delta="C6 evaluated" if report.field_available else "C6 N/A")
    m2.metric("Weighted score",   f"{report.weighted_score:.2f}",
              delta="IRA-adjusted max ≈4.3")
    m3.metric("C1 Duration",      f"{report.duration_ms:.1f} ms",
              delta=f"{report.phases_detected}-phasic · {report.discharge_type}")
    m4.metric("C3 Slope ratio",   f"{report.slope_ratio:.2f}",
              delta="rise/fall (≥1.5 or ≤0.67)")
    m5.metric("C4 Slow wave",     f"{report.slow_wave_power*100:.1f}%",
              delta="2–5 Hz delta power")
    m6.metric("C5 Suppression",   f"{report.bg_suppression_ratio:.2f}×",
              delta="post/pre RMS (<0.60)")

    st.divider()

    # ── Main layout: plot left, criteria right ────────────────────────
    left, right = st.columns([3, 1])

    with left:
        st.subheader("EEG — 500 ms window")

        half_win = int(0.25 * fs)
        w_s = max(0, ann_idx - half_win)
        w_e = min(len(eeg), ann_idx + half_win)
        seg = eeg[w_s:w_e]
        t_ms = (np.arange(len(seg)) - (ann_idx - w_s)) / fs * 1000

        seg_f = detrend(seg.astype(float), type='linear')
        seg_f -= seg_f.mean()
        seg_f = _bandpass_filter(seg_f, fs)

        fig = plt.figure(figsize=(10, 7.5), facecolor="#0e1117")
        gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)
        axes = [fig.add_subplot(gs[i]) for i in range(3)]
        for ax in axes:
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="#aaaaaa", labelsize=8)
            ax.spines[:].set_color("#2a2a3a")

        # Row 1: Signal
        axes[0].plot(t_ms, seg, color="#3a4055", lw=0.8, alpha=0.6, label="Raw")
        axes[0].plot(t_ms, seg_f, color="#4da6ff", lw=1.3, label="Filtered (0.5–70 Hz)")
        axes[0].axvline(0, color="#ff6b6b", lw=1.8, ls="--", label="Annotation")

        # BG regions (C2 ref)
        bg_ms = 100
        axes[0].axvspan(t_ms[0], t_ms[0] + bg_ms, alpha=0.10, color="#f59e0b", label="BG ref (C2)")
        axes[0].axvspan(t_ms[-1] - bg_ms, t_ms[-1], alpha=0.10, color="#f59e0b")
        # Post zone (C4/C5)
        axes[0].axvspan(0, min(200, t_ms[-1]), alpha=0.07, color="#a78bfa", label="Post-discharge (C4/C5)")

        axes[0].set_ylabel("µV", color="#aaaaaa", fontsize=9)
        axes[0].legend(fontsize=7, framealpha=0.2, facecolor="#1a1a2e",
                       labelcolor="white", loc="upper left")
        axes[0].set_title(source_label, color="#cccccc", fontsize=9, pad=4)

        # Row 2: Derivative (C3)
        deriv = np.diff(seg_f, prepend=seg_f[0])
        axes[1].plot(t_ms, deriv, color="#a78bfa", lw=0.9)
        axes[1].axvline(0, color="#ff6b6b", lw=1.8, ls="--")
        axes[1].axhline(0, color="#2a2a3a", lw=0.6)
        ira_c3_color = "#f59e0b" if report.criteria_details.get("C3_waveform_asymmetry") else "#888"
        axes[1].set_ylabel(f"d/dt  C3\n[IRA LOW]", color=ira_c3_color, fontsize=7)

        # Row 3: Running RMS (C5)
        win_r = max(4, int(0.02 * fs))
        rms_t = np.array([
            np.sqrt(np.mean(seg_f[max(0, i - win_r): i + win_r] ** 2))
            for i in range(len(seg_f))
        ])
        axes[2].plot(t_ms, rms_t, color="#22c55e", lw=0.9)
        axes[2].axvline(0, color="#ff6b6b", lw=1.8, ls="--")
        ira_c5_color = "#f59e0b" if report.criteria_details.get("C5_background_disruption") else "#888"
        axes[2].set_ylabel(f"RMS  C5\n[IRA LOW]", color=ira_c5_color, fontsize=7)
        axes[2].set_xlabel("Time relative to annotation (ms)", color="#aaaaaa", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Criteria")

        _c6r = report.c6_field_result or {}
        if _c6r:
            _c6_desc = (f"Field: {_c6r.get('field_type','?')} · "
                        f"rev={_c6r.get('polarity_reversal_count',0)} · "
                        f"smooth={_c6r.get('smoothness_score',0):.2f} · "
                        f"focus={_c6r.get('focus_channel','?')}")
        else:
            _c6_desc = "Multi-channel only — N/A"

        crit_meta = {
            "C1_diphasic_sharp_morphology": ("C1", "Di/tri-phasic morphology",
                                              "20–200 ms, ≥2 phases", "C1"),
            "C2_duration_differs_from_bg":  ("C2", "Duration ≠ background",
                                              "Half-wave ratio <0.7 or >1.5", "C2"),
            "C3_waveform_asymmetry":        ("C3", "Waveform asymmetry",
                                              "Slope ratio ≥1.5 or ≤0.67", "C3"),
            "C4_aftergoing_slow_wave":      ("C4", "After-going slow wave",
                                              ">30% delta (2–5 Hz)", "C4"),
            "C5_background_disruption":     ("C5", "Background disruption",
                                              "Post/pre RMS < 0.60", "C5"),
            "C6_physiologic_field":         ("C6", "Physiologic field",
                                              _c6_desc, "C6"),
        }

        for key, (code, name, desc, ira_key) in crit_meta.items():
            val  = report.criteria_details.get(key, False)
            ira  = IRA_LABEL.get(ira_key, "")
            if val == "n/a":
                bg, border, icon = "#1a1a2e", "#555566", "➖"
            elif val:
                bg, border, icon = "#0d2218", "#22c55e", "✅"
            else:
                bg, border, icon = "#220d0d", "#ef4444", "❌"

            st.markdown(
                f"<div style='padding:7px 10px;margin:3px 0;border-radius:7px;"
                f"border-left:3px solid {border};background:{bg}'>"
                f"<span style='font-size:12px;font-weight:600;color:{border}'>"
                f"{icon} {code} — {name}</span>"
                f"<span style='font-size:10px;color:#666;float:right'>IRA {ira}</span><br>"
                f"<span style='font-size:10px;color:#666'>{desc}</span></div>",
                unsafe_allow_html=True
            )

        # Weighted score bar
        st.markdown("<br>", unsafe_allow_html=True)
        max_ws = sum(CRITERION_IRA.values())
        pct    = min(report.weighted_score / max_ws, 1.0)
        bar_color = "#22c55e" if report.is_epileptiform_clinical else "#ef4444"
        st.markdown(
            f"<div style='font-size:11px;color:#aaa;margin-bottom:4px'>"
            f"IRA-weighted score: {report.weighted_score:.2f} / {max_ws:.2f}</div>"
            f"<div style='background:#1a1a2e;border-radius:6px;height:12px;overflow:hidden'>"
            f"<div style='background:{bar_color};width:{pct*100:.1f}%;height:100%;border-radius:6px'></div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Donut: 5 or 6 evaluable criteria
        st.markdown("<br>", unsafe_allow_html=True)
        met   = report.criteria_met
        total = 6 if report.field_available else 5
        fig2, ax2 = plt.subplots(figsize=(2.4, 2.4))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")
        pie_colors = ["#22c55e"] * met + ["#2a2a3a"] * (total - met)
        ax2.pie(
            [1] * total, colors=pie_colors, startangle=90,
            wedgeprops=dict(width=0.5, edgecolor="#0e1117", linewidth=2)
        )
        ax2.text(0, 0.08, f"{met}", ha="center", va="center",
                 fontsize=20, fontweight="bold", color="white")
        ax2.text(0, -0.25, f"/ {total}", ha="center", va="center",
                 fontsize=11, color="#aaaaaa")
        st.pyplot(fig2)
        plt.close(fig2)

        # IRA legend
        st.markdown(
            "<div style='font-size:10px;color:#666;margin-top:8px'>"
            "IRA = Inter-rater agreement<br>(Kural et al. 2020)<br>"
            "HIGH ≥0.6 · MOD 0.4–0.6 · LOW ⚠ &lt;0.4</div>",
            unsafe_allow_html=True
        )

    # ── C6: Voltage topomap ──────────────────────────────────────────
    if report.c6_field_result:
        st.divider()
        st.subheader("C6 — Voltage Field & Spike Propagation")
        fr = report.c6_field_result
        c6c1, c6c2, c6c3, c6c4 = st.columns(4)
        c6c1.metric("Field type",        fr.get("field_type","?"))
        c6c2.metric("Polarity reversals",fr.get("polarity_reversal_count",0),
                    delta="≥2 required")
        c6c3.metric("Smoothness",        f"{fr.get('smoothness_score',0):.2f}",
                    delta="0–1 (higher = smoother)")
        c6c4.metric("Focus channel",     fr.get("focus_channel","?"),
                    delta=f"latency {fr.get('peak_latency_ms',0):+.1f} ms")
        try:
            from topomap import plot_topomap, plot_delay_map
            tc1, tc2 = st.columns(2)
            with tc1:
                fig_t, ax_t = plt.subplots(figsize=(4.5, 4.5))
                fig_t.patch.set_facecolor("#0e1117")
                plot_topomap(fr, ax=ax_t, fs=fs)
                st.pyplot(fig_t); plt.close(fig_t)
            with tc2:
                fig_d, ax_d = plt.subplots(figsize=(4.5, 4.5))
                fig_d.patch.set_facecolor("#0e1117")
                plot_delay_map(fr, ax=ax_d, fs=fs)
                st.pyplot(fig_d); plt.close(fig_d)
        except Exception as e:
            st.warning(f"Topomap error: {e}")

    # ── Full recording overview ───────────────────────────────────────
    st.divider()
    st.subheader("Full Recording Overview")
    ov_len = min(len(eeg), int(fs * 30))
    t_full = np.arange(ov_len) / fs

    fig3, ax3 = plt.subplots(figsize=(12, 2.2))
    fig3.patch.set_facecolor("#0e1117")
    ax3.set_facecolor("#0e1117")
    ax3.plot(t_full, eeg[:ov_len], color="#4da6ff", lw=0.5, alpha=0.7)
    ax3.axvline(ann_idx / fs, color="#ff6b6b", lw=1.5, ls="--", label="Annotation")
    hw = int(0.25 * fs)
    ax3.axvspan((ann_idx - hw) / fs, (ann_idx + hw) / fs,
                alpha=0.15, color="#ff6b6b", label="500 ms window")
    ax3.set_xlabel("Time (s)", color="#aaaaaa", fontsize=9)
    ax3.set_ylabel("µV", color="#aaaaaa", fontsize=9)
    ax3.tick_params(colors="#aaaaaa")
    ax3.spines[:].set_color("#2a2a3a")
    ax3.legend(fontsize=8, framealpha=0.2, facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Full JSON ─────────────────────────────────────────────────────
    with st.expander("📋 Full report (JSON)"):
        import dataclasses
        st.json(dataclasses.asdict(report))

    # ── References ────────────────────────────────────────────────────
    with st.expander("📚 References"):
        st.markdown("""
**[1]** Nascimento FA, Beniczky S. *Teaching the 6 Criteria of the IFCN for Defining IEDs on EEG.*
Neurology Education 2023;2:e200073.

**[2]** Kural MA et al. *Criteria for defining interictal epileptiform discharges in EEG: A clinical validation study.*
Neurology 2020;94:e2139–e2147. DOI: 10.1212/WNL.0000000000009439

**Key finding (Kural 2020):**
- ≥4/5 criteria: accuracy 91%, sensitivity 96%, specificity 85%
- **≥5/5 criteria: accuracy 88%, sensitivity 81%, specificity 96%** (recommended for clinical use)
- Highest IRA criteria: C1 (0.75), C4 (0.65), C2 (0.63)
- Lowest IRA: C3 asymmetry (0.30), C5 background disruption (0.36)
        """)

except ImportError:
    pass
