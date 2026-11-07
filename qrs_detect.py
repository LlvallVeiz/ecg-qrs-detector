#!/usr/bin/env python3
"""Simplified Pan-Tompkins QRS detector (numpy only).

Pipeline: FFT bandpass 5-15 Hz -> derivative -> squaring ->
moving-window integration (150 ms) -> adaptive threshold with a
200 ms refractory period -> refine each detection to the local
maximum of the bandpassed signal.
"""

import numpy as np


def _bandpass_fft(x, fs, low=5.0, high=15.0):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    X[(freqs < low) | (freqs > high)] = 0.0
    return np.fft.irfft(X, n=len(x))


def detect_qrs(signal, fs=250):
    """Detect R peaks. Returns a numpy array of sample indices."""
    bp = _bandpass_fft(signal, fs)
    deriv = np.gradient(bp)
    squared = deriv ** 2

    win = max(1, int(0.150 * fs))  # 150 ms integration window
    integrated = np.convolve(squared, np.ones(win) / win, mode='same')

    # Adaptive threshold between the noise floor and the signal peaks.
    thr = integrated.mean() + 0.35 * (integrated.max() - integrated.mean())
    refractory = int(0.200 * fs)

    candidates = []
    i = 1
    while i < len(integrated) - 1:
        if integrated[i] > thr and integrated[i] >= integrated[i - 1] \
                and integrated[i] >= integrated[i + 1]:
            candidates.append(i)
            i += refractory  # skip the refractory period
        else:
            i += 1

    # Refine: the integrated peak lags the R wave; search +/-100 ms in
    # the bandpassed signal for the true local maximum magnitude.
    search = int(0.100 * fs)
    peaks = []
    for c in candidates:
        lo = max(0, c - search)
        hi = min(len(bp), c + search)
        peaks.append(lo + int(np.argmax(np.abs(bp[lo:hi]))))
    return np.array(sorted(set(peaks)), dtype=int)


def evaluate(detected, truth, fs=250, tol_ms=75):
    """Match detections to ground truth within +/- tol_ms.

    Returns dict with tp, fp, fn, sensitivity, precision.
    """
    tol = int(tol_ms * fs / 1000)
    used = set()
    tp = 0
    for d in detected:
        best, best_dist = None, tol + 1
        for k, gt in enumerate(truth):
            if k not in used and abs(int(d) - int(gt)) < best_dist:
                best, best_dist = k, abs(int(d) - int(gt))
        if best is not None:
            used.add(best)
            tp += 1
    fp = len(detected) - tp
    fn = len(truth) - tp
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'sensitivity': tp / len(truth) if len(truth) else 0.0,
        'precision': tp / len(detected) if len(detected) else 0.0,
    }


def mean_bpm(peaks, fs=250):
    """Mean heart rate (bpm) from peak sample indices."""
    if len(peaks) < 2:
        return 0.0
    rr = np.diff(peaks) / fs
    return float(60.0 / rr.mean())
