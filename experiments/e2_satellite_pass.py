#!/usr/bin/env python
"""E2 — LEO pass: SNR, Doppler, Doppler rate and feasible SFs over time;
sensitivity of the SF feasibility to the orbit altitude.

E2a  600 km zenith pass, 868 MHz, 14 dBm, satellite G_rx and NF from
     configs/base.yaml.  SNR in LOS (TR 38.811 clutter loss applies only in
     NLOS; the expected NLOS penalty is reported separately).  An SF is
     feasible at time t if PER(SF, SNR(t), |rate(t)|) ≤ 1 % in the PER table
     of the receiver model.
E2b  Zenith passes at 400–1000 km: share of the visible pass where each SF is
     feasible, for both receivers.  Compared with Doroshkin et al. (IEEE
     Access 2019), who measured that SF12 fails near zenith below ~550 km.

Run: python experiments/e2_satellite_pass.py [--plot-only]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from channel.satellite import SatellitePass, expected_extra_loss, los_probability  # noqa: E402
from netsim.config import load_config  # noqa: E402
from phy.ser_montecarlo import PerTable  # noqa: E402

RES = ROOT / "results"
SFS = [7, 8, 9, 10, 11, 12]
RECEIVERS = ("preamble", "tracking")


def los_snr(cfg: dict, st: dict, p_tx: float = 14.0) -> np.ndarray:
    sat = cfg["satellite"]
    fspl = 20 * np.log10(st["range_km"]) + 20 * math.log10(sat["frequency_hz"] / 1e6) + 32.45
    nf = -174 + 10 * math.log10(cfg["phy"]["bandwidth_hz"]) + sat["noise_figure_db"]
    return p_tx + sat["device_antenna_gain_dbi"] + sat["rx_gain_dbi"] - fspl - nf


def run() -> None:
    cfg = load_config()
    tabs = {rx: PerTable.load(rx) for rx in RECEIVERS}
    p = SatellitePass(0.0, 90.0, cfg["satellite"]["altitude_km"], cfg["satellite"]["frequency_hz"],
                      cfg["satellite"]["min_elevation_deg"])
    t = np.arange(math.ceil(p.t_rise), math.floor(p.t_set) + 1, 1.0)
    st = p.state(t)
    snr = los_snr(cfg, st)
    rows = []
    for i, ti in enumerate(t):
        r = {"t_s": ti, "elevation_deg": round(float(st["elevation_deg"][i]), 3),
             "range_km": round(float(st["range_km"][i]), 2),
             "delay_ms": round(float(st["range_km"][i]) / 299_792.458 * 1e3, 3),
             "doppler_hz": round(float(st["doppler_hz"][i]), 1),
             "doppler_rate_hz_s": round(float(st["doppler_rate_hz_s"][i]), 2),
             "snr_los_db": round(float(snr[i]), 2),
             "p_los": round(float(los_probability(st["elevation_deg"][i])), 3),
             "expected_nlos_penalty_db": round(float(expected_extra_loss(st["elevation_deg"][i])), 2)}
        for rx in RECEIVERS:
            for sf in SFS:
                r[f"feasible_{rx}_sf{sf}"] = int(tabs[rx](sf, float(snr[i]), float(st["doppler_rate_hz_s"][i])) <= 0.01)
        rows.append(r)
    write(RES / "e2_satellite_pass.csv", rows)

    alt_rows = []
    for h in [400, 450, 500, 550, 600, 650, 700, 800, 1000]:
        ph = SatellitePass(0.0, 90.0, h, cfg["satellite"]["frequency_hz"], cfg["satellite"]["min_elevation_deg"])
        tt = np.arange(math.ceil(ph.t_rise), math.floor(ph.t_set) + 1, 1.0)
        s = ph.state(tt)
        sn = los_snr(cfg, s)
        rate0 = abs(float(ph.state(0.0)["doppler_rate_hz_s"]))
        for rx in RECEIVERS:
            for sf in SFS:
                ok = np.array([tabs[rx](sf, float(a), float(b)) <= 0.01
                               for a, b in zip(sn, s["doppler_rate_hz_s"])])
                # SNR-only feasibility, to separate the Doppler effect
                ok_snr = np.array([tabs[rx](sf, float(a), 0.0) <= 0.01 for a in sn])
                alt_rows.append({"altitude_km": h, "receiver": rx, "sf": sf,
                                 "zenith_doppler_rate_hz_s": round(rate0, 1),
                                 "pass_duration_s": round(2 * ph.half_duration),
                                 "feasible_share": round(float(ok.mean()), 4),
                                 "feasible_share_snr_only": round(float(ok_snr.mean()), 4),
                                 "feasible_at_zenith": int(tabs[rx](sf, float(los_snr(cfg, ph.state(np.array([0.0])))[0]), rate0) <= 0.01)})
    write(RES / "e2b_altitude_sweep.csv", alt_rows)


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
    from phy.css import SNR_THRESHOLDS_DB
    rows = read(RES / "e2_satellite_pass.csv")
    t = np.array([float(r["t_s"]) for r in rows]) / 60
    g = lambda k: np.array([float(r[k]) for r in rows])
    fig, axes = plt.subplots(5, 1, figsize=(8.5, 8.8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.1, 1.1]})
    axes[0].plot(t, g("elevation_deg"), color="#1f77b4")
    axes[0].set_ylabel("elevation (°)")
    axes[1].plot(t, g("snr_los_db"), color="#d62728", label="LOS")
    axes[1].plot(t, g("snr_los_db") - g("expected_nlos_penalty_db"), color="#d62728", ls="--",
                 label="LOS − P(NLOS)·CL (mean)")
    for sf in (7, 10, 12):
        axes[1].axhline(SNR_THRESHOLDS_DB[sf], color="gray", ls=":", lw=1)
        axes[1].text(t[0], SNR_THRESHOLDS_DB[sf] + 0.3, f"SF{sf}", fontsize=7, color="gray")
    axes[1].set_ylabel("SNR (dB)")
    axes[1].legend(fontsize=7)
    axes[2].plot(t, g("doppler_hz") / 1e3, color="#2ca02c")
    axes[2].set_ylabel("Doppler (kHz)")
    axes[3].plot(t, g("doppler_rate_hz_s"), color="#9467bd")
    axes[3].set_ylabel("Doppler rate (Hz/s)")
    # feasible SF maps (A = preamble-only receiver, B = tracking receiver)
    fig2_axes = axes[4]
    for row_i, rx in enumerate(("preamble", "tracking")):
        for j, sf in enumerate(SFS):
            f = g(f"feasible_{rx}_sf{sf}")
            y = row_i * 7 + j
            fig2_axes.fill_between(t, y - 0.4, y + 0.4, where=f > 0, color=["#1f77b4", "#d62728"][row_i],
                                   step="mid", lw=0)
    fig2_axes.set_yticks([i * 7 + j for i in range(2) for j in range(6)])
    fig2_axes.set_yticklabels([f"{'A' if i == 0 else 'B'}:SF{sf}" for i in range(2) for sf in SFS], fontsize=6)
    fig2_axes.set_ylabel("PER ≤ 1 %\nA preamble / B tracking", fontsize=8)
    axes[-1].set_xlabel("time from closest approach (min), 600 km zenith pass, 14 dBm, LOS")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "e2_satellite_pass.png", dpi=150)
    plt.close(fig)

    rows = read(RES / "e2b_altitude_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(SFS)))
    for ax, rx in zip(axes, RECEIVERS):
        for c, sf in zip(cols, SFS):
            r = [x for x in rows if x["receiver"] == rx and int(x["sf"]) == sf]
            ax.plot([float(x["altitude_km"]) for x in r], [float(x["feasible_share"]) for x in r], "-o",
                    ms=3, color=c, label=f"SF{sf}")
        ax.axvline(550, color="gray", ls="--", lw=1)
        ax.text(555, 0.05, "Doroshkin 2019:\nSF12 fails below", fontsize=7, color="gray")
        ax.set_title({"preamble": "A: CFO from preamble only", "tracking": "B: tracking"}[rx], fontsize=9)
        ax.set_xlabel("orbit altitude (km), zenith pass")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("share of the pass with PER ≤ 1 %")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "e2b_altitude_sweep.png", dpi=150)
    plt.close(fig)
    print("  figures written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        run()
    plot()
