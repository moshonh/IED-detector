# IFCN Epileptiform Discharge Validator

Real-time Python algorithm to validate potential epileptiform discharges
at neurologist annotation points in clinical EEG recordings.

---

## Clinical Basis

Based on IFCN (International Federation of Clinical Neurophysiology) criteria
for identifying epileptiform discharges. Analyses a **500 ms window** (250 ms
before + 250 ms after the annotation mark) — wide enough to capture both the
discharge and its following slow wave, narrow enough to exclude unrelated
background activity.

---

## Project Structure

```
eeg_validator/
├── validate_discharge.py     # Core algorithm (import this in your app)
├── usage_example.py          # Integration example + 5 synthetic tests
├── requirements.txt          # Python dependencies
├── README.md
└── tests/
    └── test_validate_discharge.py   # pytest unit tests
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from validate_discharge import validate_epileptiform_discharge
import numpy as np

# eeg_buffer : 1D numpy array (one channel, µV)
# ann_idx    : sample index where the neurologist placed the annotation
# fs         : sampling frequency (e.g. 256 or 512 Hz)

report = validate_epileptiform_discharge(eeg_buffer, ann_idx, fs)

print(report)
# ✓ EPILEPTIFORM — Spike
#   Criteria met  : 5/6
#   Duration      : 54.7 ms
#   Amplitude Δ   : 6.84× background RMS
#   Slope ratio   : 2.31 (rise/fall)
#   Slow wave     : 41.2% delta power
```

---

## IFCN Criteria Checked

| # | Criterion             | Method                                      | Threshold        |
|---|-----------------------|---------------------------------------------|------------------|
| 1 | Paroxysmal amplitude  | Peak vs. RMS of first+last 100 ms           | ≥ 2.5× RMS       |
| 2 | Slope asymmetry       | Max rise slope / max fall slope             | ≥ 1.5×           |
| 3 | Duration              | Trough-to-trough via `find_peaks(-signal)`  | < 70 ms (Spike) / 70–200 ms (Sharp Wave) |
| 4 | Polarity change       | Sign flip in derivative near peak           | Present           |
| 5 | Slow wave             | Welch PSD of post-spike 200 ms              | > 30% in 2–5 Hz  |
| 6 | Peak sharpness        | Derivative energy ratio at peak             | > 3× background  |

A discharge is classified as **epileptiform** when **≥ 4 of 6** criteria are met
(configurable via `criteria_threshold` parameter).

---

## Key Implementation Details

### Edge Cases
Annotations at the start or end of a recording are handled safely.
The window is clamped to valid array bounds; `report.window_truncated`
is set to `True` when this occurs.

### DC Offset Removal
Each 500 ms segment undergoes `scipy.signal.detrend(type='linear')`
followed by mean subtraction before any amplitude measurements are taken.
This is critical for accurate amplitude ratios in short segments.

### Trough Detection
Duration is measured between the two troughs flanking the peak.
`scipy.signal.find_peaks` is applied to the **inverted signal** (`-signal`)
to locate local minima reliably.

---

## Running the Example

```bash
python usage_example.py
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Integration Notes

- Designed to run **synchronously** on every annotation event — typical
  execution is < 2 ms for a 256 Hz recording.
- Thread-safe (no global state).
- Works with any sampling frequency; `fs` is a runtime parameter.
- `DischargeReport` is a plain dataclass — JSON-serialisable with
  `dataclasses.asdict(report)`.
