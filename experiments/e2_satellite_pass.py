#!/usr/bin/env python
"""Experiment E2 -- Satellite pass analysis.

Section 3.6:
  - SNR, Doppler shift and Doppler rate over a LEO pass
  - Feasible SFs shaded over time

Outputs saved to results/e2_*.png and results/e2_*.csv.
Uses fixed random seed (42) for reproducibility.
"""

import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import csv

from phy.css import SNR_THRESHOLDS_DB, time_on_air, fft_bin_width
from channel.satellite import (
    simulate_pass, satellite_link_budget, clutter_loss,
    fspl as compute_fspl, slant_range,
)

# ═══════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

# Satellite scenario (Section 3.2 / 3.4)
H_KM = 600.0
FC = 868e6
MIN_ELEV = 10.0
DT = 1.0                  # time step (s)

# Link budget parameters
P_TX_DBM = 14.0           # max transmit power
G_TX_DBI = 2.0            # device antenna gain
G_RX_DBI = 6.0            # satellite receive antenna gain (config param)
BW = 125_000.0
NF_DB = 6.0

SFS = [7, 8, 9, 10, 11, 12]

# Colors per SF (light shading)
SF_COLORS = {7: "#1f77b4", 8: "#ff7f0e", 9: "#2ca02c",
             10: "#d62728", 11: "#9467bd", 12: "#8c564b"}

# ═══════════════════════════════════════════════════════════════════════════
# Compute SNR over the pass
# ═══════════════════════════════════════════════════════════════════════════

def compute_pass_snr(pass_data):
    """Compute SNR at each time step of the pass (no shadow fading)."""
    snr = np.zeros_like(pass_data["t_s"])
    for i in range(len(snr)):
        cl = float(clutter_loss(pass_data["elevation_deg"][i]))
        snr[i] = satellite_link_budget(
            p_tx_dbm=P_TX_DBM,
            g_tx_dbi=G_TX_DBI,
            g_rx_dbi=G_RX_DBI,
            fspl_db=pass_data["fspl_db"][i],
            shadow_db=0.0,     # deterministic pass for the figure
            clutter_db=cl,
            bw=BW,
            nf_db=NF_DB,
        )
    return snr


# ═══════════════════════════════════════════════════════════════════════════
# Feasible SFs
# ═══════════════════════════════════════════════════════════════════════════

def compute_feasible_sfs(snr_db_arr, doppler_rate_arr):
    """For each time step, return the set of SFs that can close the link.

    A SF is feasible if:
      1. SNR >= required SNR threshold
      2. Doppler drift over the packet < bin_width (conservative rule)
    """
    n = len(snr_db_arr)
    feasible = {sf: np.zeros(n, dtype=bool) for sf in SFS}

    for i in range(n):
        for sf in SFS:
            snr_ok = snr_db_arr[i] >= SNR_THRESHOLDS_DB[sf]
            # Doppler drift over one packet
            toa = time_on_air(20, sf)
            drift_hz = abs(doppler_rate_arr[i]) * toa
            bw_bin = fft_bin_width(sf)
            doppler_ok = drift_hz < bw_bin
            feasible[sf][i] = snr_ok and doppler_ok

    return feasible


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_pass(pass_data, snr_db, feasible):
    """Create a 4-panel figure of the satellite pass."""
    t_min = pass_data["t_s"] / 60.0  # convert to minutes

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # --- Panel 1: Elevation and SNR ---
    ax1 = axes[0]
    color_elev = "#1f77b4"
    ax1.plot(t_min, pass_data["elevation_deg"], color=color_elev,
             linewidth=2, label="Elevation")
    ax1.set_ylabel("Elevation (deg)", color=color_elev, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_elev)
    ax1.set_ylim(0, 95)

    ax1b = ax1.twinx()
    color_snr = "#d62728"
    ax1b.plot(t_min, snr_db, color=color_snr, linewidth=2, label="SNR")
    ax1b.set_ylabel("SNR (dB)", color=color_snr, fontsize=11)
    ax1b.tick_params(axis="y", labelcolor=color_snr)

    # Draw SNR thresholds
    for sf in [7, 12]:
        ax1b.axhline(SNR_THRESHOLDS_DB[sf], color=SF_COLORS[sf],
                      linestyle=":", alpha=0.5,
                      label=f"SF{sf} threshold")

    lines1 = ax1.get_legend_handles_labels()
    lines2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1[0] + lines2[0], lines1[1] + lines2[1],
               fontsize=8, loc="upper left")
    ax1.set_title("E2: LEO Satellite Pass (600 km, 868 MHz, zenith)",
                  fontsize=13)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Doppler shift ---
    ax2 = axes[1]
    ax2.plot(t_min, pass_data["doppler_hz"] / 1000, color="#2ca02c",
             linewidth=2)
    ax2.set_ylabel("Doppler shift (kHz)", fontsize=11)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Doppler rate ---
    ax3 = axes[2]
    ax3.plot(t_min, pass_data["doppler_rate_hz_s"], color="#9467bd",
             linewidth=2)
    ax3.set_ylabel("Doppler rate (Hz/s)", fontsize=11)
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Feasible SFs (shaded bands) ---
    ax4 = axes[3]
    for sf_idx, sf in enumerate(SFS):
        mask = feasible[sf]
        # Draw a horizontal band at y=sf where feasible
        for i in range(len(t_min)):
            if mask[i]:
                ax4.barh(sf, t_min[1] - t_min[0] if len(t_min) > 1 else 0.02,
                         left=t_min[i] - (t_min[1] - t_min[0]) / 2
                              if len(t_min) > 1 else t_min[i],
                         height=0.8, color=SF_COLORS[sf], alpha=0.6,
                         edgecolor="none")

    ax4.set_yticks(SFS)
    ax4.set_yticklabels([f"SF{sf}" for sf in SFS])
    ax4.set_ylabel("Spreading Factor", fontsize=11)
    ax4.set_xlabel("Time (minutes from closest approach)", fontsize=12)
    ax4.set_ylim(6.5, 12.5)
    ax4.grid(True, axis="x", alpha=0.3)

    # Legend for panel 4
    legend_patches = [Patch(facecolor=SF_COLORS[sf], alpha=0.6,
                            label=f"SF{sf} feasible")
                      for sf in SFS]
    ax4.legend(handles=legend_patches, fontsize=8, loc="upper right",
               ncol=3)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════════════

def save_csv(pass_data, snr_db, feasible):
    """Save pass data as CSV."""
    path = os.path.join(RESULTS_DIR, "e2_satellite_pass.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = (["t_s", "t_min", "elevation_deg", "range_km",
                   "delay_ms", "snr_db", "fspl_db",
                   "doppler_hz", "doppler_rate_hz_s"]
                  + [f"feasible_SF{sf}" for sf in SFS])
        writer.writerow(header)
        for i in range(len(pass_data["t_s"])):
            row = [
                f"{pass_data['t_s'][i]:.1f}",
                f"{pass_data['t_s'][i]/60:.3f}",
                f"{pass_data['elevation_deg'][i]:.2f}",
                f"{pass_data['range_km'][i]:.1f}",
                f"{pass_data['delay_s'][i]*1000:.3f}",
                f"{snr_db[i]:.2f}",
                f"{pass_data['fspl_db'][i]:.2f}",
                f"{pass_data['doppler_hz'][i]:.1f}",
                f"{pass_data['doppler_rate_hz_s'][i]:.2f}",
            ] + [str(int(feasible[sf][i])) for sf in SFS]
            writer.writerow(row)
    print(f"  CSV saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(SEED)

    print("=" * 65)
    print("  E2: Satellite Pass Analysis")
    print("=" * 65)
    print(f"  Altitude:   {H_KM} km")
    print(f"  Frequency:  {FC/1e6} MHz")
    print(f"  Min elev:   {MIN_ELEV} deg")
    print(f"  P_tx:       {P_TX_DBM} dBm")
    print(f"  G_tx:       {G_TX_DBI} dBi")
    print(f"  G_rx (sat): {G_RX_DBI} dBi")
    print()

    # Simulate the pass
    pass_data = simulate_pass(h_km=H_KM, f_c=FC,
                              min_elevation_deg=MIN_ELEV, dt=DT)
    duration = pass_data["t_s"][-1] - pass_data["t_s"][0]
    print(f"  Pass duration: {duration:.0f} s = {duration/60:.1f} min")
    print(f"  Points:        {len(pass_data['t_s'])}")

    # Compute SNR
    snr_db = compute_pass_snr(pass_data)
    print(f"\n  SNR range: {snr_db.min():.1f} to {snr_db.max():.1f} dB")

    # Feasible SFs
    feasible = compute_feasible_sfs(snr_db, pass_data["doppler_rate_hz_s"])
    print("\n  Feasible SF windows (minutes above min elevation):")
    for sf in SFS:
        n_feasible = feasible[sf].sum()
        pct = n_feasible / len(snr_db) * 100
        print(f"    SF{sf:2d}: {n_feasible:4d} / {len(snr_db)} points "
              f"({pct:5.1f}%)")

    # Key insight check
    print("\n  --- Key insight ---")
    mid = len(snr_db) // 2  # near zenith
    dr_zen = abs(pass_data["doppler_rate_hz_s"][mid])
    for sf in [7, 12]:
        toa = time_on_air(20, sf)
        drift = dr_zen * toa
        bw_bin = fft_bin_width(sf)
        ok = "OK" if drift < bw_bin else "FAILS"
        print(f"    SF{sf}: ToA={toa*1000:.0f}ms, "
              f"drift={drift:.0f}Hz, "
              f"bin={bw_bin:.0f}Hz -> {ok}")

    # Plot
    fig = plot_pass(pass_data, snr_db, feasible)
    fig_path = os.path.join(RESULTS_DIR, "e2_satellite_pass.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\n  Figure saved: {fig_path}")
    plt.close(fig)

    # CSV
    save_csv(pass_data, snr_db, feasible)

    print("\n" + "=" * 65)
    print("  E2 complete. Check results/ for figure and CSV.")
    print("=" * 65)
