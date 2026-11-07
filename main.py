#!/usr/bin/env python3
"""Run the full ECG demo: generate -> detect -> evaluate -> plot.

Outputs: ecg_plot.png, results.txt
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ecg_sim import generate_ecg
from qrs_detect import detect_qrs, evaluate, mean_bpm

FS = 250
TRUE_BPM = 72.0
DURATION_S = 10.0


def main():
    signal, truth = generate_ecg(duration_s=DURATION_S, fs=FS,
                                 heart_rate_bpm=TRUE_BPM)
    detected = detect_qrs(signal, fs=FS)
    ev = evaluate(detected, truth, fs=FS)
    det_bpm = mean_bpm(detected, fs=FS)

    lines = [
        'ECG QRS detection — results',
        '===========================',
        f'Sampling rate        : {FS} Hz',
        f'Duration             : {DURATION_S:.0f} s',
        f'Ground-truth beats   : {len(truth)}',
        f'Detected beats       : {len(detected)}',
        f'True positives       : {ev["tp"]}',
        f'False positives      : {ev["fp"]}',
        f'False negatives      : {ev["fn"]}',
        f'Sensitivity          : {ev["sensitivity"] * 100:.1f}%',
        f'Precision            : {ev["precision"] * 100:.1f}%',
        f'Ground-truth HR      : {TRUE_BPM:.1f} bpm (generator setting)',
        f'Detected mean HR     : {det_bpm:.2f} bpm',
        f'HR error             : {abs(det_bpm - TRUE_BPM):.2f} bpm',
        f'Detected R peaks (samples): {detected.tolist()}',
        f'Ground-truth R peaks (samples): {truth.tolist()}',
    ]
    text = '\n'.join(lines)
    print(text)
    with open('results.txt', 'w') as fh:
        fh.write(text + '\n')

    t = np.arange(len(signal)) / FS
    plt.figure(figsize=(12, 3.2))
    plt.plot(t, signal, lw=0.8, label='Synthetic ECG')
    plt.plot(t[truth], signal[truth], 'o', ms=7, fillstyle='none',
             label='Ground-truth R')
    plt.plot(t[detected], signal[detected], 'x', ms=8, mew=1.8,
             label='Detected R')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.title(f'ECG QRS detection — true {TRUE_BPM:.0f} bpm, '
              f'detected {det_bpm:.1f} bpm')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig('ecg_plot.png', dpi=130)
    print('\nSaved ecg_plot.png and results.txt')


if __name__ == '__main__':
    main()
