#!/usr/bin/env python
"""E1 — PHY validation.

E1a  SER vs SNR in AWGN for SF7–SF12: Monte Carlo against the exact
     noncoherent 2^SF-ary formula; SNR where a 20-byte packet reaches
     PER = 1 % and 10 % (analytic, all symbols full rate) vs the Semtech
     demodulation thresholds.
E1b  PER of a 20-byte packet vs Doppler rate at SNR_req + 5 dB, both
     receiver models (read from the PER tables), and the critical rate
     (PER crosses 10 %) per SF.

Run:  python experiments/e1_phy_validation.py [--plot-only] [--n-trials 2000]
Needs the PER tables (python -m phy.ser_montecarlo --receiver tracking|preamble).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phy.css import SNR_THRESHOLDS_DB, fft_bin_width, payload_symbols, time_on_air  # noqa: E402
from phy.receiver import drift_bins_at_packet_end, max_trackable_rate  # noqa: E402
from phy.ser_montecarlo import (PerTable, load_per_table, ser_awgn_analytic,  # noqa: E402
                                simulate_ser_awgn)

RES = ROOT / "results"
SFS = [7, 8, 9, 10, 11, 12]
SEED = 42


def run_e1a(n_trials: int) -> None:
    rows = []
    for sf in SFS:
        rng = np.random.default_rng([SEED, sf])
        thr = SNR_THRESHOLDS_DB[sf]
        for snr in np.arange(thr - 9.0, thr + 1.01, 0.5):
            rows.append({"sf": sf, "snr_db": round(float(snr), 2),
                         "ser_mc": simulate_ser_awgn(sf, float(snr), n_trials, rng=rng),
                         "ser_analytic": ser_awgn_analytic(sf, float(snr)), "n_trials": n_trials})
        print(f"  E1a SF{sf} done", flush=True)
    write(RES / "e1a_ser_vs_snr_awgn.csv", rows)
    th = []
    for sf in SFS:
        n = payload_symbols(20, sf)
        grid = np.arange(SNR_THRESHOLDS_DB[sf] - 6, SNR_THRESHOLDS_DB[sf] + 3, 0.01)
        ser = np.array([ser_awgn_analytic(sf, x) for x in grid[::10]])
        per = 1 - (1 - ser) ** n
        g = grid[::10]
        s1 = float(np.interp(0.01, per[::-1], g[::-1]))
        s10 = float(np.interp(0.10, per[::-1], g[::-1]))
        th.append({"sf": sf, "semtech_threshold_db": SNR_THRESHOLDS_DB[sf],
                   "snr_per1pct_analytic_db": round(s1, 2), "snr_per10pct_analytic_db": round(s10, 2),
                   "gap_per1pct_vs_threshold_db": round(s1 - SNR_THRESHOLDS_DB[sf], 2),
                   "n_payload_symbols": n})
    write(RES / "e1a_thresholds.csv", th)


def run_e1b() -> None:
    rows, crit = [], []
    for rx in ("preamble", "tracking"):
        tab = PerTable(load_per_table(rx))
        rates = tab.rates
        for sf in SFS:
            snr = SNR_THRESHOLDS_DB[sf] + 5.0
            per = [tab(sf, snr, r) for r in rates]
            for r, p in zip(rates, per):
                rows.append({"receiver": rx, "sf": sf, "snr_db": snr, "doppler_rate_hz_s": r, "per": p})
            above = [r for r, p in zip(rates, per) if p > 0.10]
            crit.append({"receiver": rx, "sf": sf,
                         "critical_rate_hz_s": above[0] if above else f">{rates[-1]:.0f}",
                         "last_ok_rate_hz_s": max([r for r, p in zip(rates, per) if p <= 0.10] or [0]),
                         "tracking_limit_theory_hz_s": round(max_trackable_rate(sf), 1),
                         "drift_bins_at_252hz_s_preamble_only": round(drift_bins_at_packet_end(sf, 252.0), 2),
                         "time_on_air_s": round(time_on_air(20, sf), 4),
                         "fft_bin_hz": round(fft_bin_width(sf), 2),
                         "n_packets_per_point": int(load_per_table(rx)["n_packets"])})
    write(RES / "e1b_per_vs_doppler.csv", rows)
    write(RES / "e1b_critical_rate.csv", crit)


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("  wrote", path.relative_to(ROOT))


def read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(SFS)))
    rows = read(RES / "e1a_ser_vs_snr_awgn.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for c, sf in zip(cols, SFS):
        r = [x for x in rows if int(x["sf"]) == sf]
        snr = [float(x["snr_db"]) for x in r]
        ax.semilogy(snr, [max(float(x["ser_mc"]), 1e-5) for x in r], "o", ms=3.5, color=c)
        ax.semilogy(snr, [max(float(x["ser_analytic"]), 1e-5) for x in r], "-", color=c, label=f"SF{sf}")
        ax.axvline(SNR_THRESHOLDS_DB[sf], color=c, ls=":", lw=1)
    ax.set_xlabel("SNR in 125 kHz (dB)")
    ax.set_ylabel("symbol error rate")
    ax.set_ylim(1e-4, 1)
    ax.grid(which="both", alpha=0.3)
    ax.set_title("E1a  AWGN: Monte Carlo (dots, %s trials) vs exact formula (lines);\n"
                 "dotted = Semtech thresholds" % rows[0]["n_trials"], fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "e1a_ser_vs_snr_awgn.png", dpi=150)
    plt.close(fig)

    rows = read(RES / "e1b_per_vs_doppler.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, rx in zip(axes, ("preamble", "tracking")):
        for c, sf in zip(cols, SFS):
            r = [x for x in rows if x["receiver"] == rx and int(x["sf"]) == sf]
            ax.plot([float(x["doppler_rate_hz_s"]) for x in r], [float(x["per"]) for x in r], "-o",
                    ms=3, color=c, label=f"SF{sf}")
        ax.axvline(252, color="gray", ls="--", lw=1)
        ax.text(256, 0.5, "600 km zenith", rotation=90, fontsize=7, color="gray")
        ax.set_title({"preamble": "receiver A: CFO from preamble only",
                      "tracking": "receiver B: decision-directed tracking"}[rx], fontsize=9)
        ax.set_xlabel("|Doppler rate| (Hz/s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PER, 20-byte packet, SNR_req + 5 dB")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "e1b_per_vs_doppler.png", dpi=150)
    plt.close(fig)
    print("  figures written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--n-trials", type=int, default=2000)
    a = ap.parse_args()
    if not a.plot_only:
        run_e1a(a.n_trials)
        run_e1b()
    plot()
