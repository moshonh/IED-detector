"""
usage_example.py
================
Demonstrates how to integrate validate_epileptiform_discharge
into a real-time EEG annotation pipeline.

Run:
    python usage_example.py
"""

import numpy as np
from validate_discharge import validate_epileptiform_discharge


# ─────────────────────────────────────────────────────────────────────
# Integration callback — called each time a doctor adds an annotation
# ─────────────────────────────────────────────────────────────────────
def on_annotation_added(
    eeg_buffer: np.ndarray,
    sample_idx: int,
    fs: float,
    channel_name: str = "unknown"
) -> None:
    """
    Callback to run the IFCN validator whenever a manual annotation is placed.

    Parameters
    ----------
    eeg_buffer   : Full EEG recording (1D, one channel, µV)
    sample_idx   : Index of the annotation in eeg_buffer
    fs           : Sampling frequency (Hz)
    channel_name : Label for display (e.g. 'F3', 'T4')
    """
    report = validate_epileptiform_discharge(eeg_buffer, sample_idx, fs)

    print(f"\n{'='*55}")
    print(f"Channel: {channel_name}  |  Sample: {sample_idx}  |  fs: {fs} Hz")
    print('='*55)
    print(report)
    print('='*55)

    return report


# ─────────────────────────────────────────────────────────────────────
# Test 1 — Classic Spike-and-Slow-Wave complex
# ─────────────────────────────────────────────────────────────────────
def test_spike_slow_wave(fs: float = 256) -> None:
    print("\n[TEST 1] Classic Spike + Slow Wave complex")

    np.random.seed(42)
    duration = 10   # seconds
    t = np.arange(0, duration, 1 / fs)
    eeg = np.random.normal(0, 5, len(t))   # background noise (µV)

    ann_idx = int(5 * fs)   # annotation at t = 5 s

    # Inject asymmetric spike (faster rise, slower fall)
    spike_half = int(0.03 * fs)
    spike_t    = np.linspace(-0.03, 0.03, spike_half * 2)
    spike_env  = np.exp(-((spike_t / 0.012) ** 2))
    spike_wave = 80 * spike_env * np.where(spike_t < 0, 1.5, 1.0)

    s_start = ann_idx - spike_half
    s_end   = ann_idx + spike_half
    if s_start >= 0 and s_end <= len(eeg):
        eeg[s_start:s_end] += spike_wave

    # Inject slow wave (3 Hz) for 200 ms after spike
    sw_len = int(0.2 * fs)
    sw_t   = np.linspace(0, 0.2, sw_len)
    if ann_idx + sw_len <= len(eeg):
        eeg[ann_idx: ann_idx + sw_len] += 40 * np.sin(2 * np.pi * 3 * sw_t)

    on_annotation_added(eeg, ann_idx, fs, channel_name="F3")


# ─────────────────────────────────────────────────────────────────────
# Test 2 — Sharp Wave (70–200 ms duration)
# ─────────────────────────────────────────────────────────────────────
def test_sharp_wave(fs: float = 256) -> None:
    print("\n[TEST 2] Sharp Wave (wider morphology)")

    np.random.seed(7)
    t   = np.arange(0, 10, 1 / fs)
    eeg = np.random.normal(0, 5, len(t))

    ann_idx    = int(5 * fs)
    sharp_half = int(0.06 * fs)   # ~120 ms total → Sharp Wave range
    sharp_t    = np.linspace(-0.06, 0.06, sharp_half * 2)
    sharp_wave = 60 * np.exp(-((sharp_t / 0.025) ** 2))

    s_start = ann_idx - sharp_half
    s_end   = ann_idx + sharp_half
    if s_start >= 0 and s_end <= len(eeg):
        eeg[s_start:s_end] += sharp_wave

    on_annotation_added(eeg, ann_idx, fs, channel_name="T4")


# ─────────────────────────────────────────────────────────────────────
# Test 3 — Normal EEG (should return False)
# ─────────────────────────────────────────────────────────────────────
def test_normal_activity(fs: float = 256) -> None:
    print("\n[TEST 3] Normal background EEG (expected: NOT epileptiform)")

    np.random.seed(99)
    t   = np.arange(0, 10, 1 / fs)
    eeg = np.random.normal(0, 8, len(t))   # just noise

    ann_idx = int(5 * fs)
    on_annotation_added(eeg, ann_idx, fs, channel_name="O1")


# ─────────────────────────────────────────────────────────────────────
# Test 4 — Edge case: annotation at the very start of recording
# ─────────────────────────────────────────────────────────────────────
def test_edge_start(fs: float = 256) -> None:
    print("\n[TEST 4] Edge case — annotation at recording start (sample 10)")

    np.random.seed(1)
    eeg     = np.random.normal(0, 5, int(10 * fs))
    ann_idx = 10   # only 10 samples before start

    on_annotation_added(eeg, ann_idx, fs, channel_name="Fp1")


# ─────────────────────────────────────────────────────────────────────
# Test 5 — Higher sampling rate (512 Hz)
# ─────────────────────────────────────────────────────────────────────
def test_high_fs(fs: float = 512) -> None:
    print(f"\n[TEST 5] High sampling rate ({fs} Hz)")

    np.random.seed(13)
    t   = np.arange(0, 10, 1 / fs)
    eeg = np.random.normal(0, 5, len(t))

    ann_idx   = int(5 * fs)
    half      = int(0.025 * fs)
    spike_t   = np.linspace(-0.025, 0.025, half * 2)
    spike_env = 90 * np.exp(-((spike_t / 0.010) ** 2))

    s_start = ann_idx - half
    s_end   = ann_idx + half
    if s_start >= 0 and s_end <= len(eeg):
        eeg[s_start:s_end] += spike_env

    sw_len = int(0.2 * fs)
    sw_t   = np.linspace(0, 0.2, sw_len)
    if ann_idx + sw_len <= len(eeg):
        eeg[ann_idx: ann_idx + sw_len] += 35 * np.sin(2 * np.pi * 4 * sw_t)

    on_annotation_added(eeg, ann_idx, fs, channel_name="C3")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_spike_slow_wave()
    test_sharp_wave()
    test_normal_activity()
    test_edge_start()
    test_high_fs()
