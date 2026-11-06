#!/usr/bin/env python
"""E3 — Scalability without anomalies: PDR and energy per delivered packet vs
number of devices (100–1000), every policy, 5 seeds.  Terrestrial cell only
(configs/e3_scalability.yaml), 2 h, ADR warm-started at its fixed point.

Also the pure-ALOHA sanity check of the simulator (S = G·e^{−2G}).

Run:
    python experiments/e3_scalability.py                    # baselines + ALOHA check
    python experiments/e3_scalability.py --policies LLM-Agent --backend ollama --model qwen2.5:3b
    python experiments/e3_scalability.py --plot-only
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from common import (BASELINES, COLORS, STYLES, RESULTS, add_llm_args, append_rows, done_keys,
                    llm_overrides, load_config, mean_std, read_rows, run_one)
from netsim.simulator import NetworkSimulator
from phy.css import time_on_air

CSV = RESULTS / "e3_scalability.csv"
ALOHA = RESULTS / "e3_aloha_check.csv"
N_LIST = [100, 200, 400, 600, 800, 1000]


def run(policies, n_list, seeds, args) -> None:
    for n in n_list:
        cfg = load_config("e3_scalability", overrides={"network": {"n_devices": n},
                                                       "agent": llm_overrides(args)})
        done = done_keys(CSV, ("policy", "n_devices", "seed", "model_key"))
        for seed in seeds or cfg["seeds"]:
            for pol in policies:
                mkey = cfg["agent"]["model"] if pol.startswith("LLM-Agent") else "-"
                if (pol, str(n), str(seed), mkey) in done:
                    continue
                row, _ = run_one(cfg, seed, pol, "e3", f"n{n}", args.replay_only)
                row["model_key"] = mkey
                append_rows(CSV, [row])
                print(f"  N={n:5d} {pol:16s} seed={seed} pdr={row['pdr']:.3f} "
                      f"E={row['energy_per_delivered_mj']:.1f} mJ", flush=True)


class _NoControl:
    """Every device at SF7 / 14 dBm, no adaptation (for the ALOHA check)."""
    name = "aloha"
    per_packet = False

    def attach(self, sim):
        from netsim.devices import TxParams
        for d in sim.devices:
            d.base = TxParams(7, 14, (), "terrestrial")

    def on_uplink(self, *a):
        pass

    def on_tick(self, *a):
        pass


def aloha_check() -> None:
    """Equal power (no capture), no PER, near-Poisson arrivals: compare the
    per-channel throughput with S = G·e^{−2G}."""
    rows = []
    toa = time_on_air(20, 7)
    for n in [1000, 3000, 6000, 10000, 16000, 24000, 32000]:
        cfg = load_config("e3_scalability", overrides={
            "duration_s": 3600, "network": {"n_devices": n, "report_jitter_frac": 1.0,
                                            "capture_threshold_db": 1e9},
            "deployment": {"core_radius_km": 0.2}, "terrestrial": {"shadowing_std_db": 0.0,
                                                                    "fast_fading_std_db": 0.0}})
        sim = NetworkSimulator(cfg, 0, _NoControl(), per_lookup=lambda sf, snr, r: 0.0)
        m = sim.run()
        n_ch = len(cfg["network"]["channels_mhz"])
        G = m["sent"] * toa / (cfg["duration_s"] * n_ch)
        S = m["delivered"] * toa / (cfg["duration_s"] * n_ch)
        rows.append({"n_devices": n, "G": G, "S_sim": S, "S_theory": G * math.exp(-2 * G),
                     "pdr_sim": m["pdr"], "pdr_theory": math.exp(-2 * G)})
        print(f"  ALOHA N={n:6d} G={G:.3f} S_sim={S:.4f} S_theory={G*math.exp(-2*G):.4f}", flush=True)
    import csv
    with ALOHA.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_rows(CSV)
    pols = [p for p in list(COLORS) if any(r["policy"] == p for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for pol in pols:
        ns = sorted({r["n_devices"] for r in rows if r["policy"] == pol})
        for ax, key in zip(axes[:2], ("pdr", "energy_per_delivered_mj")):
            m = [mean_std(r[key] for r in rows if r["policy"] == pol and r["n_devices"] == n) for n in ns]
            ax.errorbar(ns, [x[0] for x in m], yerr=[x[1] for x in m], fmt="o", ms=3, capsize=2,
                        ls=STYLES.get(pol, "-"), lw=1.6 if pol in STYLES else 1.2,
                        color=COLORS[pol], label=f"{pol} (n={m[0][2]} seeds)")
    axes[0].set_ylabel("PDR")
    axes[1].set_ylabel("energy per delivered packet (mJ)")
    axes[1].set_yscale("log")
    for ax in axes[:2]:
        ax.set_xlabel("number of devices")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    if ALOHA.exists():
        a = read_rows(ALOHA)
        g = np.linspace(0, max(r["G"] for r in a) * 1.05, 200)
        axes[2].plot(g, g * np.exp(-2 * g), "k-", label=r"theory $G\,e^{-2G}$")
        axes[2].plot([r["G"] for r in a], [r["S_sim"] for r in a], "o", color="#d62728", label="simulator")
        axes[2].set_xlabel("offered load G per channel")
        axes[2].set_ylabel("throughput S")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)
        axes[2].set_title("pure-ALOHA sanity check (SF7, no capture)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "e3_scalability.png", dpi=150)
    plt.close(fig)
    print("  figure written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", default=BASELINES)
    ap.add_argument("--n", nargs="+", type=int, default=N_LIST)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--skip-aloha", action="store_true")
    add_llm_args(ap)
    args = ap.parse_args()
    if not args.plot_only:
        if not args.skip_aloha and not ALOHA.exists():
            aloha_check()
        run(args.policies, args.n, args.seeds, args)
    plot()
