"""
topomap.py — Voltage map & C6 field validation using MNE
=========================================================
Requires: mne >= 1.0

Install:
    pip install mne

C6 criterion (Kural 2020):
  "Distribution of negative and positive potentials on the scalp
   suggests a source in the brain, corresponding to a radial,
   oblique, or tangential orientation of the source."

Implementation:
  1. Build MNE Info with standard_1020 montage
  2. Find spike peak latency per channel (delay map)
  3. Plot voltage topomap at spike peak (mne.viz.plot_topomap)
  4. Check polarity reversal between adjacent electrode pairs
  5. Check spatial smoothness of the interpolated field
  6. Classify field: Radial / Tangential / Oblique / Non-physiologic

Falls back to scipy RBF interpolation if MNE is not installed.
"""

import numpy as np
import warnings

# ─────────────────────────────────────────────────────────────────────
# Standard 10-20 adjacency for polarity-reversal check
# ─────────────────────────────────────────────────────────────────────
ADJACENT_PAIRS = [
    ("Fp1","F3"), ("Fp2","F4"), ("F3","C3"), ("F4","C4"),
    ("F7","T3"),  ("F8","T4"),  ("C3","P3"), ("C4","P4"),
    ("T3","T5"),  ("T4","T6"),  ("P3","O1"), ("P4","O2"),
    ("F3","Fz"),  ("F4","Fz"),  ("C3","Cz"), ("C4","Cz"),
    ("P3","Pz"),  ("P4","Pz"),  ("Fz","Cz"), ("Cz","Pz"),
]

# 2D positions for RBF fallback (no MNE)
ELECTRODE_POS_2D = {
    "Fp1": (-0.18,  0.85), "Fp2": ( 0.18,  0.85),
    "F7":  (-0.52,  0.60), "F3":  (-0.28,  0.50),
    "Fz":  ( 0.00,  0.50), "F4":  ( 0.28,  0.50), "F8":  ( 0.52,  0.60),
    "T3":  (-0.72,  0.00), "C3":  (-0.38,  0.00),
    "Cz":  ( 0.00,  0.00),
    "C4":  ( 0.38,  0.00), "T4":  ( 0.72,  0.00),
    "T5":  (-0.52, -0.60), "P3":  (-0.28, -0.50),
    "Pz":  ( 0.00, -0.50), "P4":  ( 0.28, -0.50), "T6":  ( 0.52, -0.60),
    "O1":  (-0.18, -0.85), "O2":  ( 0.18, -0.85),
}


def _mne_available():
    try:
        import mne  # noqa: F401
        return True
    except ImportError:
        return False


def _build_mne_info(ch_names: list, fs: float):
    """Build MNE Info with standard_1020 montage. Returns (info, valid_chs, valid_idx)."""
    import mne
    mne.set_log_level("ERROR")
    montage    = mne.channels.make_standard_montage("standard_1020")
    montage_chs = set(montage.ch_names)
    valid       = [(i, ch) for i, ch in enumerate(ch_names) if ch in montage_chs]
    if len(valid) < 3:
        raise ValueError(f"Need ≥3 valid 10-20 channels. Found: {[c for _,c in valid]}")
    valid_idx  = [i for i, _ in valid]
    valid_chs  = [ch for _, ch in valid]
    info = mne.create_info(ch_names=valid_chs, sfreq=fs, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    return info, valid_chs, valid_idx


def _find_peak(seg: np.ndarray, local_ann: int, fs: float, search_ms=60.0) -> int:
    radius = int(search_ms / 1000.0 * fs)
    s = max(0, local_ann - radius)
    e = min(len(seg), local_ann + radius)
    return s + int(np.argmax(np.abs(seg[s:e])))


# ─────────────────────────────────────────────────────────────────────
# C6 core analysis
# ─────────────────────────────────────────────────────────────────────
def analyse_field(eeg_matrix: np.ndarray, ch_names: list,
                  ann_idx: int, fs: float) -> dict:
    """
    Analyse voltage field at spike peak for C6.

    Parameters
    ----------
    eeg_matrix : (n_ch, n_samples) µV
    ch_names   : channel name list
    ann_idx    : annotation sample index
    fs         : sampling frequency Hz

    Returns dict with: passed, field_type, focus_channel, peak_latency_ms,
    polarity_reversal_count, smoothness_score, voltage_at_peak, delay_map,
    focus_ratio, mne_available
    """
    from scipy.signal import detrend, butter, filtfilt

    USE_MNE = _mne_available()

    half_win  = int(0.25 * fs)
    start     = max(0, ann_idx - half_win)
    end       = min(eeg_matrix.shape[1], ann_idx + half_win)
    local_ann = ann_idx - start

    # Filter each channel
    windows = {}
    for i, ch in enumerate(ch_names):
        if ch not in ELECTRODE_POS_2D:
            continue
        seg = eeg_matrix[i, start:end].astype(float)
        seg = detrend(seg, type="linear")
        seg -= seg.mean()
        if len(seg) >= 27:
            nyq = fs / 2.0
            b, a = butter(4, [0.5/nyq, min(70.0/nyq, 0.99)], btype="band")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seg = filtfilt(b, a, seg)
        windows[ch] = seg

    if len(windows) < 3:
        return {"passed": False, "field_type": "Insufficient channels",
                "focus_channel": None, "peak_latency_ms": 0.0,
                "polarity_reversal_count": 0, "smoothness_score": 0.0,
                "voltage_at_peak": {}, "delay_map": {},
                "focus_ratio": 0.0, "mne_available": USE_MNE}

    # Peak per channel
    peak_idx  = {ch: _find_peak(seg, local_ann, fs) for ch, seg in windows.items()}
    peak_v    = {ch: float(windows[ch][peak_idx[ch]]) for ch in windows}
    delay_map = {ch: round((peak_idx[ch] - local_ann) / fs * 1000.0, 2)
                 for ch in windows}

    focus_ch        = max(peak_v, key=lambda c: abs(peak_v[c]))
    peak_latency_ms = delay_map[focus_ch]
    fp              = peak_idx[focus_ch]

    voltage_at_peak = {ch: float(windows[ch][min(fp, len(windows[ch])-1)])
                       for ch in windows}

    # Polarity reversal
    threshold = abs(peak_v[focus_ch]) * 0.10
    reversal_count = sum(
        1 for a, b in ADJACENT_PAIRS
        if a in voltage_at_peak and b in voltage_at_peak
        and abs(voltage_at_peak[a]) > threshold
        and abs(voltage_at_peak[b]) > threshold
        and np.sign(voltage_at_peak[a]) != np.sign(voltage_at_peak[b])
    )

    # Smoothness via MNE or RBF
    smoothness_score = 0.0
    if USE_MNE:
        try:
            import mne, matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            info, valid_chs, _ = _build_mne_info(list(windows.keys()), fs)
            v_arr = np.array([voltage_at_peak[c] for c in valid_chs])
            fig, ax = plt.subplots(figsize=(3, 3))
            mne.viz.plot_topomap(v_arr, info, axes=ax, show=False,
                                 contours=0, sensors=False,
                                 extrapolate="head", sphere=0.09)
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3].astype(float)
            plt.close(fig)
            gy, gx = np.gradient(img.mean(axis=2))
            gs = float(np.std(np.sqrt(gx**2+gy**2)))
            smoothness_score = float(np.clip(1.0 - gs/(gs+8.0), 0, 1))
        except Exception:
            smoothness_score = 0.0
    else:
        try:
            from scipy.interpolate import RBFInterpolator
            chl = list(voltage_at_peak.keys())
            xa  = np.array([ELECTRODE_POS_2D[c][0] for c in chl])
            ya  = np.array([ELECTRODE_POS_2D[c][1] for c in chl])
            va  = np.array([voltage_at_peak[c] for c in chl])
            rbf = RBFInterpolator(np.column_stack([xa,ya]), va,
                                  kernel="thin_plate_spline")
            xi = np.linspace(-1,1,60)
            gx, gy = np.meshgrid(xi, xi)
            gz = rbf(np.column_stack([gx.ravel(),gy.ravel()])).reshape(60,60)
            gy2, gx2 = np.gradient(gz)
            gs = float(np.std(np.sqrt(gx2**2+gy2**2)))
            smoothness_score = float(np.clip(1.0 - gs/(gs+5.0), 0, 1))
        except Exception:
            smoothness_score = 0.0

    # Field classification
    focus_v   = abs(peak_v[focus_ch])
    other_max = max((abs(v) for c,v in peak_v.items() if c!=focus_ch), default=0.0)
    focus_ratio = focus_v / other_max if other_max > 1e-9 else 0.0

    if reversal_count >= 2 and smoothness_score >= 0.50:
        field_type = "Radial" if focus_ratio >= 2.0 else \
                     "Oblique" if focus_ratio >= 1.2 else "Tangential"
    else:
        field_type = "Non-physiologic"

    return {
        "passed":                   bool(field_type in ("Radial","Tangential","Oblique")),
        "field_type":               field_type,
        "focus_channel":            focus_ch,
        "peak_latency_ms":          peak_latency_ms,
        "polarity_reversal_count":  reversal_count,
        "smoothness_score":         round(smoothness_score, 3),
        "voltage_at_peak":          voltage_at_peak,
        "delay_map":                delay_map,
        "focus_ratio":              round(focus_ratio, 3),
        "mne_available":            USE_MNE,
    }


# ─────────────────────────────────────────────────────────────────────
# Topomap plot (MNE preferred, RBF fallback)
# ─────────────────────────────────────────────────────────────────────
def _draw_head(ax):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((0,0),1.0,fill=False,edgecolor="#cccccc",lw=1.8,zorder=5))
    ax.plot([-.07,0,.07],[1.00,1.12,1.00],color="#cccccc",lw=1.8,zorder=5)
    for s in (-1,1):
        ax.plot([s*.98,s*1.08,s*1.08,s*.98],[.10,.06,-.06,-.10],
                color="#cccccc",lw=1.5,zorder=5)
    ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.30)
    ax.set_aspect("equal"); ax.axis("off")


def plot_topomap(field_result: dict, ax=None,
                 title="Voltage at spike peak",
                 vlim=None, cmap="RdBu_r", fs=256.0):
    """2D voltage topomap. Uses MNE if available, RBF otherwise."""
    import matplotlib.pyplot as plt
    vap = field_result.get("voltage_at_peak", {})
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5,4.5))
        fig.patch.set_facecolor("#0e1117")
    else:
        fig = ax.figure
    ax.set_facecolor("#0e1117")

    if not vap:
        ax.text(.5,.5,"No data",transform=ax.transAxes,
                ha="center",va="center",color="white")
        return fig, ax

    ch_list = list(vap.keys())
    v_arr   = np.array([vap[c] for c in ch_list])
    if vlim is None:
        vlim = max(float(np.percentile(np.abs(v_arr),98)), 1.0)

    delay_map = field_result.get("delay_map", {})
    focus_ch  = field_result.get("focus_channel","")
    USE_MNE   = field_result.get("mne_available", False)
    backend   = "MNE" if USE_MNE else "RBF"
    plotted   = False

    if USE_MNE:
        try:
            import mne
            mne.set_log_level("ERROR")
            info, valid_chs, _ = _build_mne_info(ch_list, fs)
            v_valid = np.array([vap[c] for c in valid_chs])
            im, _ = mne.viz.plot_topomap(
                v_valid, info, axes=ax, show=False,
                cmap=cmap, vlim=(-vlim, vlim),
                contours=6, sensors=True,
                extrapolate="head", sphere=0.09,
                names=valid_chs,
            )
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
            cb.set_label("µV", color="#aaaaaa", fontsize=8)
            cb.ax.tick_params(colors="#aaaaaa", labelsize=7)
            # Delay labels
            try:
                from mne.channels.layout import _find_topomap_coords
                coords = _find_topomap_coords(info, picks="eeg")
                for i, ch in enumerate(valid_chs):
                    xp, yp = coords[i]
                    d = delay_map.get(ch, 0.0)
                    fc = "#ffdd44" if ch==focus_ch else "#ffffff"
                    ax.text(xp, yp+0.025, f"{d:+.0f}ms",
                            ha="center", va="bottom",
                            fontsize=8 if ch==focus_ch else 6,
                            color=fc, zorder=12,
                            bbox=dict(boxstyle="round,pad=0.08",
                                      fc="#00000077", ec="none"))
            except Exception:
                pass
            plotted = True
        except Exception:
            plotted = False

    if not plotted:
        try:
            from scipy.interpolate import RBFInterpolator
            valid = [c for c in ch_list if c in ELECTRODE_POS_2D]
            xa = np.array([ELECTRODE_POS_2D[c][0] for c in valid])
            ya = np.array([ELECTRODE_POS_2D[c][1] for c in valid])
            va = np.array([vap[c] for c in valid])
            rbf = RBFInterpolator(np.column_stack([xa,ya]), va,
                                  kernel="thin_plate_spline")
            xi = np.linspace(-1.1,1.1,120)
            gx, gy = np.meshgrid(xi, xi)
            gz = rbf(np.column_stack([gx.ravel(),gy.ravel()])).reshape(120,120)
            mask = (gx**2+gy**2)>1.0
            gz_m = np.ma.array(gz, mask=mask)
            im = ax.contourf(gx,gy,gz_m,levels=60,
                             cmap=cmap,vmin=-vlim,vmax=vlim)
            ax.contour(gx,gy,gz_m,levels=6,colors="k",linewidths=0.4,alpha=0.35)
            cb = fig.colorbar(im,ax=ax,fraction=0.035,pad=0.03)
            cb.set_label("µV",color="#aaaaaa",fontsize=8)
            cb.ax.tick_params(colors="#aaaaaa",labelsize=7)
            for c in valid:
                xp,yp = ELECTRODE_POS_2D[c]
                dot = "#ff4444" if vap[c]>0 else "#4488ff"
                sz  = 70 if c==focus_ch else 30
                ax.scatter(xp,yp,s=sz,c=dot,zorder=10,
                           edgecolors="white",linewidths=0.5)
                d = delay_map.get(c,0.0)
                ax.text(xp,yp+0.10,f"{c}\n{d:+.0f}ms",
                        ha="center",va="bottom",fontsize=5.5,
                        color="white",zorder=11,
                        bbox=dict(boxstyle="round,pad=0.1",
                                  fc="#00000066",ec="none"))
            _draw_head(ax)
        except Exception as e:
            ax.text(.5,.5,f"Plot error:\n{e}",
                    transform=ax.transAxes,ha="center",
                    va="center",color="white",fontsize=7)

    ft      = field_result.get("field_type","")
    rev     = field_result.get("polarity_reversal_count",0)
    sm      = field_result.get("smoothness_score",0)
    passed  = field_result.get("passed",False)
    color   = "#22c55e" if passed else "#ef4444"
    icon    = "✓" if passed else "✗"
    ax.set_title(
        f"{title}  [{backend}]\n"
        f"{icon} C6: {ft}  |  rev={rev}  smooth={sm:.2f}",
        color=color, fontsize=8, pad=6
    )
    return fig, ax


def plot_delay_map(field_result: dict, ax=None,
                   title="Spike propagation delay (ms)", fs=256.0):
    """2D delay topomap. MNE preferred, RBF fallback."""
    import matplotlib.pyplot as plt
    delay_map = field_result.get("delay_map",{})
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5,4.5))
        fig.patch.set_facecolor("#0e1117")
    else:
        fig = ax.figure
    ax.set_facecolor("#0e1117")

    if not delay_map:
        ax.text(.5,.5,"No data",transform=ax.transAxes,
                ha="center",va="center",color="white")
        return fig, ax

    ch_list = list(delay_map.keys())
    d_arr   = np.array(list(delay_map.values()))
    vlim_d  = max(float(np.percentile(np.abs(d_arr),95)), 5.0)
    USE_MNE = field_result.get("mne_available", False)
    plotted = False

    if USE_MNE:
        try:
            import mne
            mne.set_log_level("ERROR")
            info, valid_chs, _ = _build_mne_info(ch_list, fs)
            d_valid = np.array([delay_map[c] for c in valid_chs])
            im, _ = mne.viz.plot_topomap(
                d_valid, info, axes=ax, show=False,
                cmap="RdBu_r", vlim=(-vlim_d, vlim_d),
                contours=4, sensors=True,
                extrapolate="head", sphere=0.09,
                names=valid_chs,
            )
            cb = fig.colorbar(im,ax=ax,fraction=0.035,pad=0.03)
            cb.set_label("ms",color="#aaaaaa",fontsize=8)
            cb.ax.tick_params(colors="#aaaaaa",labelsize=7)
            plotted = True
        except Exception:
            plotted = False

    if not plotted:
        try:
            from scipy.interpolate import RBFInterpolator
            valid = [c for c in ch_list if c in ELECTRODE_POS_2D]
            xa = np.array([ELECTRODE_POS_2D[c][0] for c in valid])
            ya = np.array([ELECTRODE_POS_2D[c][1] for c in valid])
            da = np.array([delay_map[c] for c in valid])
            rbf = RBFInterpolator(np.column_stack([xa,ya]), da,
                                  kernel="thin_plate_spline")
            xi = np.linspace(-1.1,1.1,100)
            gx, gy = np.meshgrid(xi, xi)
            gz = rbf(np.column_stack([gx.ravel(),gy.ravel()])).reshape(100,100)
            mask = (gx**2+gy**2)>1.0
            gz_m = np.ma.array(gz, mask=mask)
            im = ax.contourf(gx,gy,gz_m,levels=40,
                             cmap="coolwarm",vmin=-vlim_d,vmax=vlim_d)
            ax.contour(gx,gy,gz_m,levels=4,colors="k",linewidths=0.3,alpha=0.3)
            cb = fig.colorbar(im,ax=ax,fraction=0.035,pad=0.03)
            cb.set_label("ms",color="#aaaaaa",fontsize=8)
            cb.ax.tick_params(colors="#aaaaaa",labelsize=7)
            focus_ch = field_result.get("focus_channel","")
            for c in valid:
                xp,yp = ELECTRODE_POS_2D[c]
                d = delay_map[c]
                ax.scatter(xp,yp,s=80 if c==focus_ch else 35,
                           c=[d],cmap="coolwarm",vmin=-vlim_d,vmax=vlim_d,
                           zorder=10,edgecolors="white",linewidths=0.5)
                ax.text(xp,yp+0.10,f"{c}\n{d:+.0f}ms",
                        ha="center",va="bottom",fontsize=5.5,
                        color="white",zorder=11,
                        bbox=dict(boxstyle="round,pad=0.1",
                                  fc="#00000066",ec="none"))
            _draw_head(ax)
        except Exception as e:
            ax.text(.5,.5,f"Plot error:\n{e}",
                    transform=ax.transAxes,ha="center",
                    va="center",color="white",fontsize=7)

    backend = "MNE" if USE_MNE else "RBF"
    ax.set_title(f"{title}  [{backend}]",color="#cccccc",fontsize=8,pad=6)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────
# Synthetic test helper
# ─────────────────────────────────────────────────────────────────────
def make_synthetic_multichannel(fs=256, duration=10.0,
                                 focus_ch="C3", spike_amp=80.0,
                                 seed=42):
    """
    Generate 19-channel synthetic EEG with a physiologic spike focus.
    Returns (eeg_matrix [19 × N], ch_names, ann_idx).
    """
    rng      = np.random.default_rng(seed)
    ch_names = list(ELECTRODE_POS_2D.keys())
    n_samp   = int(duration * fs)
    ann_idx  = int(5 * fs)
    eeg      = rng.normal(0, 5, (len(ch_names), n_samp))
    fpos     = np.array(ELECTRODE_POS_2D[focus_ch])

    for i, ch in enumerate(ch_names):
        cpos      = np.array(ELECTRODE_POS_2D[ch])
        dist      = np.linalg.norm(cpos - fpos)
        amp       = spike_amp / (1 + 3*dist**2)
        if cpos[0] * fpos[0] < -0.1:
            amp *= -0.4
        delay_s   = int(dist * 0.002 * fs)
        rise_s    = int(0.005 * fs)
        fall_s    = int(0.025 * fs)
        wave      = np.concatenate([np.linspace(0,amp,rise_s),
                                    np.linspace(amp,0,fall_s)])
        ws = ann_idx + delay_s - rise_s
        we = ws + len(wave)
        if ws >= 0 and we <= n_samp:
            eeg[i, ws:we] += wave

    return eeg, ch_names, ann_idx
