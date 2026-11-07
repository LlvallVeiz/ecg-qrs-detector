#!/usr/bin/env python3
"""Synthetic ECG generator.

Builds a ~10 s ECG-like signal at 250 Hz with a parameterised heart
rate. Each beat is a sum of Gaussian bumps approximating PQRST
morphology. Returns the signal plus the ground-truth R-peak sample
indices so detector accuracy can be measured honestly.
"""

import numpy as np

# (offset_s relative to R, amplitude mV, width sigma s)
_WAVES = [
    (-0.160, 0.12, 0.025),   # P wave
    (-0.020, -0.10, 0.008),  # Q dip
    (0.000, 1.00, 0.010),    # R peak
    (0.020, -0.22, 0.009),   # S dip
    (0.300, 0.28, 0.035),    # T wave
]


def generate_ecg(duration_s=10.0, fs=250, heart_rate_bpm=72.0,
                 noise_std=0.02, seed=42):
    """Generate a synthetic ECG.

    Returns (signal, r_peaks) where r_peaks is a numpy array of the
    ground-truth R-peak sample indices.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    signal = np.zeros(n)

    rr_mean = 60.0 / heart_rate_bpm
    beat_times = []
    bt = 0.6  # first beat after 0.6 s
    while bt < duration_s - 0.5:
        beat_times.append(bt)
        bt += rr_mean * (1.0 + rng.normal(0, 0.02))  # ~2% HRV jitter

    r_peaks = []
    for bt in beat_times:
        for off, amp, width in _WAVES:
            signal += amp * np.exp(-0.5 * ((t - (bt + off)) / width) ** 2)
        r_peaks.append(int(round(bt * fs)))

    signal += 0.05 * np.sin(2 * np.pi * 0.3 * t)          # baseline wander
    signal += rng.normal(0, noise_std, n)                   # measurement noise
    return signal, np.array(r_peaks, dtype=int)
