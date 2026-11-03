#!/usr/bin/env python
"""Experiment E1 — LoRa PHY validation.

Section 3.6:
  - SER vs SNR per SF in AWGN (curves should shift ~2.5 dB per SF step)
  - SER vs Doppler rate per SF

Outputs saved to results/e1_*.png and results/e1_*.csv.
Uses fixed random seed (42) for reproducibility.
"""

import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv

from phy.css import SNR_THRESHOLDS_DB, fft_bin_width, time_on_air
from phy.ser_montecarlo import simulate_ser_awgn, simulate_ser_doppler

# ═══════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
SFS = [7, 8, 9, 10, 11, 12]
N_TRIALS = 500           # per (SF, SNR) point — increase for smoother curves
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

# Figure 1: SER vs SNR in AWGN
SNR_RANGE = np.arange(-25, 5, 0.5)  # dB

# Figure 2: SER vs Doppler rate (at SNR = threshold + 5 dB for each SF)
DOPPLER_RATES = np.array([0, 25, 50, 100, 150, 200, 275, 400, 600, 800, 1000],
                         dtype=float)  # Hz/s
SNR_MARGIN_DB = 5.0  # above threshold — so we see the Doppler effect, not noise

# Colors per SF
COLORS = {7: "#1f77b4", 8: "#ff7f0e", 9: "#2ca02c",
          10: "#d62728", 11: "#9467bd", 12: "#8c564b"}

# ═══════════════════════════════════════════════════════════════════════════
# E1a — SER vs SNR (AWGN)
# ═══════════════════════════════════════════════════════════════════════════

def run_ser_vs_snr():
    """Monte Carlo SER vs SNR for SF7-SF12 in AWGN."""
    print("=" * 65)
    print("  E1a: SER vs SNR (AWGN), N_trials =", N_TRIALS)
    print("=" * 65)

    rng = np.random.default_rng(SEED)
    results = {}

    for sf in SFS:
        ser_list = []
        for snr_db in SNR_RANGE:
            ser = simulate_ser_awgn(sf, snr_db, n_trials=N_TRIALS, rng=rng)
            ser_list.append(ser)
        results[sf] = np.array(ser_list)
        # Print threshold check
        thr = SNR_THRESHOLDS_DB[sf]
        idx = np.argmin(np.abs(SNR_RANGE - thr))
        print(f"  SF{sf:2d}  threshold={thr:+6.1f} dB  "
              f"SER@threshold={results[sf][idx]:.4f}")

    return results


def plot_ser_vs_snr(results):
    """Plot and save SER vs SNR figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for sf in SFS:
        label = f"SF{sf}"
        ax.semilogy(SNR_RANGE, np.clip(results[sf], 1e-4, 1.0),
                     color=COLORS[sf], linewidth=1.8, label=label)
        # Mark the threshold
        thr = SNR_THRESHOLDS_DB[sf]
        ax.axvline(thr, color=COLORS[sf], linestyle=":", alpha=0.4)

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Symbol Error Rate (SER)", fontsize=12)
    ax.set_title("E1a: LoRa SER vs SNR in AWGN (Monte Carlo)", fontsize=13)
    ax.set_xlim(SNR_RANGE[0], SNR_RANGE[-1])
    ax.set_ylim(1e-4, 1.0)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "e1a_ser_vs_snr_awgn.png")
    fig.savefig(path, dpi=150)
    print(f"\n  Figure saved: {path}")
    plt.close(fig)
    return path


def save_csv_snr(results):
    """Save SER vs SNR data as CSV."""
    path = os.path.join(RESULTS_DIR, "e1a_ser_vs_snr_awgn.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["snr_db"] + [f"SER_SF{sf}" for sf in SFS]
        writer.writerow(header)
        for i, snr_db in enumerate(SNR_RANGE):
            row = [f"{snr_db:.1f}"] + [f"{results[sf][i]:.6f}" for sf in SFS]
            writer.writerow(row)
    print(f"  CSV saved:    {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# E1b — SER vs Doppler rate
# ═══════════════════════════════════════════════════════════════════════════

def run_ser_vs_doppler():
    """Monte Carlo SER vs Doppler rate for SF7-SF12."""
    print()
    print("=" * 65)
    print("  E1b: SER vs Doppler rate, N_trials =", N_TRIALS)
    print("=" * 65)

    rng = np.random.default_rng(SEED)
    results = {}

    for sf in SFS:
        snr_db = SNR_THRESHOLDS_DB[sf] + SNR_MARGIN_DB
        bw = fft_bin_width(sf)
        print(f"\n  SF{sf:2d}  SNR={snr_db:+.1f} dB  "
              f"(threshold{SNR_MARGIN_DB:+.0f})  "
              f"bin_width={bw:.1f} Hz")

        ser_list = []
        for fd_rate in DOPPLER_RATES:
            ser = simulate_ser_doppler(sf, snr_db, fd_rate,
                                       n_trials=N_TRIALS, rng=rng)
            ser_list.append(ser)
            if fd_rate in [0, 275, 1000]:
                print(f"    f_D'={fd_rate:6.0f} Hz/s  SER={ser:.4f}")

        results[sf] = np.array(ser_list)

    return results


def plot_ser_vs_doppler(results):
    """Plot and save SER vs Doppler rate figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for sf in SFS:
        label = (f"SF{sf} "
                 f"(SNR={SNR_THRESHOLDS_DB[sf]+SNR_MARGIN_DB:+.0f} dB, "
                 f"bin={fft_bin_width(sf):.0f} Hz)")
        ax.semilogy(DOPPLER_RATES, np.clip(results[sf], 1e-4, 1.0),
                     "o-", color=COLORS[sf], linewidth=1.8,
                     markersize=4, label=label)

    # Mark the typical LEO Doppler rate at zenith
    ax.axvline(275, color="gray", linestyle="--", alpha=0.6,
               label="LEO zenith ~275 Hz/s")

    ax.set_xlabel("Doppler rate (Hz/s)", fontsize=12)
    ax.set_ylabel("Symbol Error Rate (SER)", fontsize=12)
    ax.set_title("E1b: LoRa SER vs Doppler Rate "
                 "(AWGN + Doppler, Monte Carlo)", fontsize=13)
    ax.set_xlim(0, DOPPLER_RATES[-1])
    ax.set_ylim(1e-4, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "e1b_ser_vs_doppler.png")
    fig.savefig(path, dpi=150)
    print(f"\n  Figure saved: {path}")
    plt.close(fig)
    return path


def save_csv_doppler(results):
    """Save SER vs Doppler data as CSV."""
    path = os.path.join(RESULTS_DIR, "e1b_ser_vs_doppler.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["doppler_rate_hz_s"] + [f"SER_SF{sf}" for sf in SFS]
        writer.writerow(header)
        for i, fd_rate in enumerate(DOPPLER_RATES):
            row = [f"{fd_rate:.0f}"] + [f"{results[sf][i]:.6f}" for sf in SFS]
            writer.writerow(row)
    print(f"  CSV saved:    {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # E1a: SER vs SNR
    res_snr = run_ser_vs_snr()
    plot_ser_vs_snr(res_snr)
    save_csv_snr(res_snr)

    # E1b: SER vs Doppler rate
    res_doppler = run_ser_vs_doppler()
    plot_ser_vs_doppler(res_doppler)
    save_csv_doppler(res_doppler)

    print("\n" + "=" * 65)
    print("  E1 complete. Check results/ for figures and CSV files.")
    print("=" * 65)
