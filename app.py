"""
app.py — Streamlit UI for IFCN Epileptiform Discharge Validator
===============================================================
Run:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal as sp_signal
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from validate_discharge import validate_epileptiform_discharge, DischargeReport

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IFCN EEG Validator",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 IFCN Epileptiform Discharge Validator")
st.caption("500 ms window (250 ms before + 250 ms after annotation) · IFCN clinical criteria")

# ─────────────────────────────────────────────
# Sidebar — signal controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Signal Settings")

    fs = st.selectbox("Sampling frequency (Hz)", [256, 512, 1024], index=0)

    st.divider()
    st.subheader("Synthetic EEG Generator")

    noise_std   = st.slider("Background noise (µV)",   1.0, 20.0, 5.0, 0.5)
    spike_amp   = st.slider("Spike amplitude (µV)",    0.0, 150.0, 80.0, 5.0)
    spike_width = st.slider("Spike half-width (ms)",   5.0, 100.0, 25.0, 1.0)
    sw_amp      = st.slider("Slow wave amplitude (µV)", 0.0, 80.0, 40.0, 5.0)
    sw_freq     = st.slider("Slow wave frequency (Hz)", 1.0, 8.0, 3.0, 0.5)
    seed        = st.number_input("Random seed", 0, 9999, 42, 1)

    st.divider()
    st.subheader("Classifier")
    threshold = st.slider("Criteria threshold (out of 6)", 1, 6, 4, 1)

    st.divider()
    st.caption("Upload your own EEG below ↓")

# ─────────────────────────────────────────────
# Signal source: upload or synthetic
# ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload EEG channel (.npy or .csv, single channel, µV)",
    type=["npy", "csv"],
    help="NumPy array (.npy) or single-column CSV of raw EEG samples."
)

@st.cache_data
def make_synthetic(fs, noise_std, spike_amp, spike_width_ms,
                   sw_amp, sw_freq, seed):
    rng = np.random.default_rng(seed)
    duration = 10   # seconds
    t   = np.arange(0, duration, 1 / fs)
    eeg = rng.normal(0, noise_std, len(t))

    ann_idx = int(5 * fs)

    # Asymmetric spike
    hw     = int((spike_width_ms / 1000) * fs)
    if hw > 0:
        spike_t = np.linspace(-spike_width_ms/1000, spike_width_ms/1000, hw * 2)
        sigma   = (spike_width_ms / 1000) * 0.45
        wave    = spike_amp * np.exp(-((spike_t / sigma) ** 2)) * np.where(spike_t < 0, 1.5, 1.0)
        s, e    = ann_idx - hw, ann_idx + hw
        if s >= 0 and e <= len(eeg):
            eeg[s:e] += wave

    # Slow wave
    sw_len = int(0.2 * fs)
    if sw_amp > 0 and ann_idx + sw_len <= len(eeg):
        sw_t = np.linspace(0, 0.2, sw_len)
        eeg[ann_idx: ann_idx + sw_len] += sw_amp * np.sin(2 * np.pi * sw_freq * sw_t)

    return eeg, ann_idx

if uploaded is not None:
    try:
        if uploaded.name.endswith(".npy"):
            eeg = np.load(uploaded).astype(float).ravel()
        else:
            import pandas as pd
            eeg = pd.read_csv(uploaded, header=None).iloc[:, 0].to_numpy(dtype=float)

        default_ann = len(eeg) // 2
        ann_idx = st.slider(
            "Annotation position (sample index)",
            int(fs * 0.5), len(eeg) - int(fs * 0.5),
            default_ann, 1
        )
        source_label = f"Uploaded — {uploaded.name}"
    except Exception as ex:
        st.error(f"Could not load file: {ex}")
        st.stop()
else:
    eeg, ann_idx = make_synthetic(
        fs, noise_std, spike_amp, spike_width, sw_amp, sw_freq, int(seed)
    )
    source_label = "Synthetic EEG"

# ─────────────────────────────────────────────
# Run validator
# ─────────────────────────────────────────────
report: DischargeReport = validate_epileptiform_discharge(
    eeg, ann_idx, fs, criteria_threshold=threshold
)

# ─────────────────────────────────────────────
# Verdict banner
# ─────────────────────────────────────────────
if report.is_epileptiform:
    st.success(
        f"✅ **EPILEPTIFORM DISCHARGE** — {report.discharge_type}  "
        f"({report.criteria_met}/{6} criteria met)",
        icon="⚡"
    )
else:
    st.warning(
        f"⬜ **NOT epileptiform**  "
        f"({report.criteria_met}/{6} criteria met)",
        icon="🔍"
    )

if report.window_truncated:
    st.info(
        f"⚠️ Window truncated — annotation near recording boundary "
        f"({report.window_samples} samples used)",
        icon="⚠️"
    )

# ─────────────────────────────────────────────
# Metrics row
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Duration",       f"{report.duration_ms:.1f} ms",
            delta=report.discharge_type if report.discharge_type != "None" else "–")
col2.metric("Amplitude ratio", f"{report.amplitude_ratio:.2f}×",
            delta="vs background RMS")
col3.metric("Slope ratio",     f"{report.slope_ratio:.2f}",
            delta="rise / fall")
col4.metric("Slow wave power", f"{report.slow_wave_power * 100:.1f}%",
            delta="2–5 Hz band")

st.divider()

# ─────────────────────────────────────────────
# Two-column layout: EEG plot | Criteria
# ─────────────────────────────────────────────
left, right = st.columns([3, 1])

# ── EEG plot ─────────────────────────────────
with left:
    st.subheader("EEG — 500 ms window")

    half_win = int(0.25 * fs)
    start    = max(0, ann_idx - half_win)
    end      = min(len(eeg), ann_idx + half_win)
    segment  = eeg[start:end]
    t_ms     = (np.arange(len(segment)) - (ann_idx - start)) / fs * 1000  # ms relative to annotation

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333333")

    # Raw + filtered
    filtered = segment.copy().astype(float)
    from scipy.signal import detrend
    filtered = detrend(filtered, type='linear')
    filtered -= filtered.mean()
    b, a = sp_signal.butter(4, [0.5 / (fs/2), min(70.0 / (fs/2), 0.99)], btype='band')
    from scipy.signal import filtfilt
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered = filtfilt(b, a, filtered) if len(filtered) >= 27 else filtered

    axes[0].plot(t_ms, segment, color="#444a5a", linewidth=0.8, alpha=0.5, label="Raw")
    axes[0].plot(t_ms, filtered, color="#4da6ff", linewidth=1.2, label="Filtered (0.5–70 Hz)")
    axes[0].axvline(0, color="#ff6b6b", linewidth=1.5, linestyle="--", label="Annotation")

    # Shade pre/post windows used by criteria
    bg_len_ms = 100
    axes[0].axvspan(t_ms[0],  t_ms[0] + bg_len_ms,  alpha=0.08, color="#ffaa00", label="BG region")
    axes[0].axvspan(t_ms[-1] - bg_len_ms, t_ms[-1], alpha=0.08, color="#ffaa00")

    axes[0].set_ylabel("Amplitude (µV)", color="#aaaaaa", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.3,
                   facecolor="#1a1a2e", labelcolor="white")
    axes[0].set_title(source_label, color="#cccccc", fontsize=10, pad=6)

    # Derivative
    deriv = np.diff(filtered, prepend=filtered[0])
    axes[1].plot(t_ms, deriv, color="#a78bfa", linewidth=0.9, label="d/dt")
    axes[1].axvline(0, color="#ff6b6b", linewidth=1.5, linestyle="--")
    axes[1].axhline(0, color="#333333", linewidth=0.5)
    axes[1].set_ylabel("Derivative", color="#aaaaaa", fontsize=9)
    axes[1].set_xlabel("Time relative to annotation (ms)", color="#aaaaaa", fontsize=10)
    axes[1].legend(loc="upper left", fontsize=8, framealpha=0.3,
                   facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Criteria panel ────────────────────────────
with right:
    st.subheader("IFCN Criteria")

    criteria_labels = {
        "C1_paroxysmal_amplitude": ("C1", "Paroxysmal amplitude",  "Peak ≥ 2.5× BG RMS"),
        "C2_slope_asymmetry":      ("C2", "Slope asymmetry",       "Rise ≥ 1.5× fall speed"),
        "C3_duration_valid":       ("C3", "Duration",              "< 70 ms or 70–200 ms"),
        "C4_polarity_change":      ("C4", "Polarity change",       "Sign flip at peak"),
        "C5_slow_wave_present":    ("C5", "Slow wave",             "> 30% delta power (2–5 Hz)"),
        "C6_peak_sharpness":       ("C6", "Peak sharpness",        "> 3× BG energy"),
    }

    for key, (code, name, desc) in criteria_labels.items():
        val = report.criteria_details.get(key, False)
        icon  = "✅" if val else "❌"
        color = "#22c55e" if val else "#ef4444"
        st.markdown(
            f"<div style='padding:8px 10px; margin:4px 0; border-radius:8px; "
            f"border-left: 3px solid {color}; background:#1a1a2e'>"
            f"<span style='font-size:13px; font-weight:600; color:{color}'>{icon} {code} — {name}</span><br>"
            f"<span style='font-size:11px; color:#888'>{desc}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Mini donut via matplotlib
    st.markdown("<br>", unsafe_allow_html=True)
    met   = report.criteria_met
    unmet = 6 - met
    fig2, ax2 = plt.subplots(figsize=(2.4, 2.4))
    fig2.patch.set_facecolor("#0e1117")
    ax2.set_facecolor("#0e1117")
    colors = ["#22c55e"] * met + ["#333344"] * unmet
    wedges, _ = ax2.pie(
        [1]*6, colors=colors, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="#0e1117", linewidth=2)
    )
    ax2.text(0, 0, f"{met}/6", ha="center", va="center",
             fontsize=16, fontweight="bold", color="white")
    st.pyplot(fig2)
    plt.close(fig2)

# ─────────────────────────────────────────────
# Full-recording overview
# ─────────────────────────────────────────────
st.divider()
st.subheader("Full Recording Overview")

overview_len = min(len(eeg), int(fs * 30))   # show max 30 s
t_full = np.arange(overview_len) / fs

fig3, ax3 = plt.subplots(figsize=(12, 2.2))
fig3.patch.set_facecolor("#0e1117")
ax3.set_facecolor("#0e1117")
ax3.plot(t_full, eeg[:overview_len], color="#4da6ff", linewidth=0.5, alpha=0.7)
ax3.axvline(ann_idx / fs, color="#ff6b6b", linewidth=1.5, linestyle="--", label="Annotation")
ax3.axvspan((ann_idx - half_win) / fs, (ann_idx + half_win) / fs,
            alpha=0.15, color="#ff6b6b", label="500 ms window")
ax3.set_xlabel("Time (s)", color="#aaaaaa", fontsize=9)
ax3.set_ylabel("µV", color="#aaaaaa", fontsize=9)
ax3.tick_params(colors="#aaaaaa")
ax3.spines[:].set_color("#333333")
ax3.legend(fontsize=8, framealpha=0.3, facecolor="#1a1a2e", labelcolor="white")
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# ─────────────────────────────────────────────
# Raw report JSON (expandable)
# ─────────────────────────────────────────────
with st.expander("📋 Full report (JSON)"):
    import dataclasses, json
    st.json(dataclasses.asdict(report))
