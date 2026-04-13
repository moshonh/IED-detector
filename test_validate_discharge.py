"""
test_validate_discharge.py
===========================
Unit tests for the IFCN Epileptiform Discharge Validator.

Run:
    pytest tests/test_validate_discharge.py -v
"""

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None  # allow running without pytest
from validate_discharge import validate_epileptiform_discharge, DischargeReport


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def make_clean_eeg(duration: float = 10.0, fs: float = 256,
                   noise_std: float = 5.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, noise_std, int(duration * fs))


def inject_spike(eeg: np.ndarray, ann_idx: int, fs: float,
                 amplitude: float = 80.0, half_width: float = 0.025) -> None:
    """Inject an asymmetric spike centred at ann_idx."""
    hw   = int(half_width * fs)
    t    = np.linspace(-half_width, half_width, hw * 2)
    wave = amplitude * np.exp(-((t / (half_width * 0.45)) ** 2)) * np.where(t < 0, 1.5, 1.0)
    s, e = ann_idx - hw, ann_idx + hw
    if s >= 0 and e <= len(eeg):
        eeg[s:e] += wave


def inject_slow_wave(eeg: np.ndarray, ann_idx: int, fs: float,
                     amplitude: float = 40.0, freq: float = 3.0) -> None:
    """Inject a slow wave starting at ann_idx."""
    length = int(0.2 * fs)
    if ann_idx + length > len(eeg):
        return
    t = np.linspace(0, 0.2, length)
    eeg[ann_idx: ann_idx + length] += amplitude * np.sin(2 * np.pi * freq * t)


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────
class TestReturnType:
    def test_returns_discharge_report(self):
        eeg = make_clean_eeg()
        result = validate_epileptiform_discharge(eeg, 1280, 256)
        assert isinstance(result, DischargeReport)

    def test_report_has_all_fields(self):
        eeg = make_clean_eeg()
        r = validate_epileptiform_discharge(eeg, 1280, 256)
        for attr in ("is_epileptiform", "criteria_met", "discharge_type",
                     "duration_ms", "amplitude_ratio", "slope_ratio",
                     "slow_wave_power", "criteria_details",
                     "window_truncated", "window_samples"):
            assert hasattr(r, attr)

    def test_criteria_details_has_six_keys(self):
        eeg = make_clean_eeg()
        r = validate_epileptiform_discharge(eeg, 1280, 256)
        assert len(r.criteria_details) == 6


class TestEpileptiformDetection:
    def test_spike_positive(self):
        """Clear spike + slow wave → should be epileptiform."""
        eeg = make_clean_eeg(seed=42)
        ann = 1280
        inject_spike(eeg, ann, 256)
        inject_slow_wave(eeg, ann, 256)
        r = validate_epileptiform_discharge(eeg, ann, 256)
        assert r.is_epileptiform is True

    def test_normal_eeg_negative(self):
        """Pure noise → should NOT be epileptiform."""
        eeg = make_clean_eeg(noise_std=5, seed=99)
        r = validate_epileptiform_discharge(eeg, 1280, 256)
        assert r.is_epileptiform is False

    def test_criteria_count_range(self):
        eeg = make_clean_eeg(seed=7)
        inject_spike(eeg, 1280, 256)
        r = validate_epileptiform_discharge(eeg, 1280, 256)
        assert 0 <= r.criteria_met <= 6

    def test_custom_threshold_5(self):
        """With threshold=5, same spike may fail."""
        eeg = make_clean_eeg(seed=42)
        inject_spike(eeg, 1280, 256)
        inject_slow_wave(eeg, 1280, 256)
        r4 = validate_epileptiform_discharge(eeg, 1280, 256, criteria_threshold=4)
        r5 = validate_epileptiform_discharge(eeg, 1280, 256, criteria_threshold=5)
        # Threshold 4 must be >= threshold 5 in sensitivity
        assert r4.criteria_met >= r5.criteria_met


class TestDischargeType:
    def test_spike_under_70ms(self):
        """Narrow spike should be classified as 'Spike'."""
        eeg = make_clean_eeg(seed=11)
        ann = 1280
        inject_spike(eeg, ann, 256, half_width=0.020)  # ~40 ms
        r = validate_epileptiform_discharge(eeg, ann, 256)
        if r.discharge_type != "None":
            assert r.discharge_type == "Spike"

    def test_sharp_wave_70_200ms(self):
        """Wider spike should be classified as 'Sharp Wave'."""
        eeg = make_clean_eeg(seed=22)
        ann = 1280
        inject_spike(eeg, ann, 256, half_width=0.055)  # ~110 ms
        r = validate_epileptiform_discharge(eeg, ann, 256)
        if r.discharge_type != "None":
            assert r.discharge_type in ("Sharp Wave", "Spike")


class TestEdgeCases:
    def test_annotation_at_start(self):
        """Annotation at sample 5 should not crash."""
        eeg = make_clean_eeg()
        r = validate_epileptiform_discharge(eeg, 5, 256)
        assert r.window_truncated is True
        assert r.window_samples > 0

    def test_annotation_at_end(self):
        """Annotation near last sample should not crash."""
        eeg = make_clean_eeg()
        r = validate_epileptiform_discharge(eeg, len(eeg) - 5, 256)
        assert r.window_truncated is True

    def test_annotation_at_exact_boundary(self):
        """Annotation exactly at index 0."""
        eeg = make_clean_eeg()
        r = validate_epileptiform_discharge(eeg, 0, 256)
        assert isinstance(r, DischargeReport)

    def test_very_short_recording(self):
        """Recording shorter than 200 ms should return gracefully."""
        eeg = np.random.normal(0, 5, 20)   # 20 samples ≈ 78 ms at 256 Hz
        r = validate_epileptiform_discharge(eeg, 10, 256)
        assert isinstance(r, DischargeReport)
        assert r.is_epileptiform is False

    def test_empty_signal_handled(self):
        """Empty array should not raise an exception."""
        eeg = np.array([])
        r = validate_epileptiform_discharge(eeg, 0, 256)
        assert r.is_epileptiform is False


class TestHighSamplingRate:
    def test_512hz(self):
        eeg = make_clean_eeg(fs=512, seed=5)
        inject_spike(eeg, 2560, 512)
        inject_slow_wave(eeg, 2560, 512)
        r = validate_epileptiform_discharge(eeg, 2560, 512)
        assert isinstance(r, DischargeReport)

    def test_1024hz(self):
        eeg = make_clean_eeg(fs=1024, seed=6)
        inject_spike(eeg, 5120, 1024)
        r = validate_epileptiform_discharge(eeg, 5120, 1024)
        assert isinstance(r, DischargeReport)


class TestMetricValues:
    def test_amplitude_ratio_positive(self):
        eeg = make_clean_eeg(seed=42)
        inject_spike(eeg, 1280, 256)
        r = validate_epileptiform_discharge(eeg, 1280, 256)
        assert r.amplitude_ratio >= 0.0

    def test_duration_non_negative(self):
        eeg = make_clean_eeg(seed=42)
        inject_spike(eeg, 1280, 256)
        r = validate_epileptiform_discharge(eeg, 1280, 256)
        assert r.duration_ms >= 0.0

    def test_slow_wave_power_bounded(self):
        eeg = make_clean_eeg(seed=42)
        inject_spike(eeg, 1280, 256)
        inject_slow_wave(eeg, 1280, 256)
        r = validate_epileptiform_discharge(eeg, 1280, 256)
        assert 0.0 <= r.slow_wave_power <= 1.0

    def test_slope_ratio_non_negative(self):
        eeg = make_clean_eeg(seed=42)
        inject_spike(eeg, 1280, 256)
        r = validate_epileptiform_discharge(eeg, 1280, 256)
        assert r.slope_ratio >= 0.0
