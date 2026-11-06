#!/usr/bin/env python
"""E4 — Anomaly scenarios S1–S5: PDR over time around each event, recovery
time, energy, for every policy, 5 seeds (paired: common random numbers).

Run (resumable; finished runs are skipped):
    python experiments/e4_anomalies.py                       # baselines only
    python experiments/e4_anomalies.py --policies LLM-Agent \
        --backend ollama --model qwen2.5:3b                   # the agent
    python experiments/e4_anomalies.py --plot-only           # figures from CSV

Outputs: results/e4_summary.csv, results/e4_timeseries.csv,
         results/e4_pdr_timeline.png, results/e4_event_bars.png,
         results/traces/e4/*.jsonl (agent decisions)
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from common import (AGENTS, BASELINES, COLORS, STYLES, RESULTS, add_llm_args, append_rows, done_keys,
                    llm_overrides, load_config, mean_std, read_rows, run_one)
from netsim.config import SCENARIOS

SUMMARY = RESULTS / "e4_summary.csv"
TIMESERIES = RESULTS / "e4_timeseries.csv"
LABELS = {"s1_jamming": "S1 jamming", "s2_deep_fade": "S2 deep fade",
          "s3_pass_end": "S3 end of satellite pass", "s4_gateway_outage": "S4 gateway outage",
          "s5_traffic_surge": "S5 traffic surge"}


def run(policies: list[str], scenarios: list[str], seeds: list[int] | None, args) -> None:
    for scen in scenarios:
        cfg = load_config(scen, overrides={"agent": llm_overrides(args)})
        model = cfg["agent"]["model"]
        done = done_keys(SUMMARY, ("scenario", "policy", "seed", "model_key"))
        for seed in seeds or cfg["seeds"]:
            for pol in policies:
                mkey = model if pol.startswith("LLM-Agent") else "-"
                if (scen, pol, str(seed), mkey) in done:
                    continue
                row, ts = run_one(cfg, seed, pol, "e4", scen, args.replay_only)
                row["model_key"] = mkey
                append_rows(SUMMARY, [row])
                append_rows(TIMESERIES, [{"scenario": scen, "policy": pol, "seed": seed,
                                          "model_key": mkey, "t_start": float(t), "sent": float(s),
                                          "delivered": float(d), "energy_j": float(e)}
                                         for t, s, d, e in zip(ts["t_start"], ts["sent"],
                                                               ts["delivered"], ts["energy_j"])])
                print(f"  {scen:18s} {pol:24s} seed={seed} pdr={row['pdr']:.3f} "
                      f"pdr_event={row.get('pdr_event', float('nan')):.3f} "
                      f"rec={row.get('recovery_time_s')} wall={row['wall_s']}s", flush=True)


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts_rows = read_rows(TIMESERIES)
    rows = read_rows(SUMMARY)
    pols = [p for p in BASELINES + AGENTS[:1] if any(r["policy"] == p for r in rows)]
    pols = [p for p in pols if p != "Fixed-SF12"]          # off-scale energy, saturates ALOHA
    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(9, 2.6 * len(SCENARIOS)), sharex=True)
    for ax, scen in zip(axes, SCENARIOS):
        cfg = load_config(scen)
        a = cfg["anomalies"][0]
        t0, t1 = a["start_s"], a["start_s"] + a.get("duration_s", 0)
        if t1 > t0:
            ax.axvspan(t0 / 60, t1 / 60, color="#f2c6c6", alpha=0.6, lw=0)
        else:
            ax.axvline(t0 / 60, color="#d62728", lw=1.5, ls="--")
        from channel.satellite import SatellitePass
        sat = cfg["satellite"]
        for p in sat.get("passes", []) if sat.get("enabled", True) else []:
            tmp = SatellitePass(0.0, p["max_elevation_deg"], sat["altitude_km"], sat["frequency_hz"],
                                sat["min_elevation_deg"])
            tc = p.get("t_closest_s", p.get("t_set_s", 0) - tmp.half_duration)
            ax.axvspan((tc - tmp.half_duration) / 60, (tc + tmp.half_duration) / 60,
                       color="#9ecae1", alpha=0.5, lw=0)
        for pol in pols:
            sel = [r for r in ts_rows if r["scenario"] == scen and r["policy"] == pol]
            if not sel:
                continue
            ts = sorted({r["t_start"] for r in sel})
            pdr = []
            for t in ts:
                s = sum(r["sent"] for r in sel if r["t_start"] == t)
                d = sum(r["delivered"] for r in sel if r["t_start"] == t)
                pdr.append(d / s if s else np.nan)
            n_seeds = len({r["seed"] for r in sel})
            ax.plot((np.array(ts) + 0.5 * cfg["metrics"]["pdr_bin_s"]) / 60, pdr, marker="o", ms=2.5,
                    ls=STYLES.get(pol, "-"), lw=1.8 if pol in STYLES else 1.3, color=COLORS[pol],
                    label=f"{pol} (n={n_seeds} seeds)")
        ax.set_ylabel("PDR")
        ax.set_title(LABELS[scen], fontsize=10, loc="left")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3, loc="lower left")
    axes[-1].set_xlabel("time (min); 10-min bins pooled over seeds; red = event, blue = satellite pass")
    fig.tight_layout()
    fig.savefig(RESULTS / "e4_pdr_timeline.png", dpi=150)
    plt.close(fig)

    # wide version for the two-column report: 1 x 5 panels, shared y axis
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(16, 3.3), sharey=True)
    for ax, scen in zip(axes, SCENARIOS):
        cfg = load_config(scen)
        a = cfg["anomalies"][0]
        t0, t1 = a["start_s"], a["start_s"] + a.get("duration_s", 0)
        if t1 > t0:
            ax.axvspan(t0 / 60, t1 / 60, color="#f2c6c6", alpha=0.6, lw=0)
        else:
            ax.axvline(t0 / 60, color="#d62728", lw=1.5, ls="--")
        from channel.satellite import SatellitePass
        sat = cfg["satellite"]
        for p in sat.get("passes", []) if sat.get("enabled", True) else []:
            tmp = SatellitePass(0.0, p["max_elevation_deg"], sat["altitude_km"], sat["frequency_hz"],
                                sat["min_elevation_deg"])
            tc = p.get("t_closest_s", p.get("t_set_s", 0) - tmp.half_duration)
            ax.axvspan((tc - tmp.half_duration) / 60, (tc + tmp.half_duration) / 60,
                       color="#9ecae1", alpha=0.5, lw=0)
        for pol in pols:
            sel = [r for r in ts_rows if r["scenario"] == scen and r["policy"] == pol]
            if not sel:
                continue
            ts = sorted({r["t_start"] for r in sel})
            pdr = []
            for t in ts:
                s_ = sum(r["sent"] for r in sel if r["t_start"] == t)
                d_ = sum(r["delivered"] for r in sel if r["t_start"] == t)
                pdr.append(d_ / s_ if s_ else np.nan)
            ax.plot((np.array(ts) + 0.5 * cfg["metrics"]["pdr_bin_s"]) / 60, pdr, marker="o", ms=2,
                    ls=STYLES.get(pol, "-"), lw=1.6 if pol in STYLES else 1.1, color=COLORS[pol], label=pol)
        ax.set_title(LABELS[scen], fontsize=10)
        ax.set_xlabel("time (min)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PDR (10-min bins, 5 seeds pooled)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(pols), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RESULTS / "e4_pdr_timeline_wide.png", dpi=170)
    plt.close(fig)

    # bars: PDR during the event window and energy per delivered packet
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(SCENARIOS))
    w = 0.8 / max(1, len(pols))
    for i, pol in enumerate(pols):
        m, s = [], []
        e_m, e_s = [], []
        for scen in SCENARIOS:
            sel = [r for r in rows if r["scenario"] == scen and r["policy"] == pol]
            mm, ss, _ = mean_std(r["pdr_event"] for r in sel)
            m.append(mm); s.append(ss)
            mm, ss, _ = mean_std(r["energy_per_delivered_event_mj"] for r in sel)
            e_m.append(mm); e_s.append(ss)
        axes[0].bar(x + i * w, m, w, yerr=s, color=COLORS[pol], label=pol, capsize=2)
        axes[1].bar(x + i * w, e_m, w, yerr=e_s, color=COLORS[pol], capsize=2)
    for ax, yl in zip(axes, ("PDR in event window", "energy / delivered packet (mJ)\nin event window")):
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([LABELS[s].split(" ", 1)[0] for s in SCENARIOS])
        ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_yscale("log")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(pols), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS / "e4_event_bars.png", dpi=150)
    plt.close(fig)
    print("  figures written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", default=BASELINES)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true")
    add_llm_args(ap)
    args = ap.parse_args()
    if not args.plot_only:
        run(args.policies, args.scenarios, args.seeds, args)
    plot()
