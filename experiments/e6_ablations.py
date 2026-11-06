#!/usr/bin/env python
"""E6 — Ablations of the agent on S1–S5, 5 seeds, same model as E4:
    LLM-Agent-noAnomalyTool   get_anomaly_report removed from the tool list
    LLM-Agent-noFacts         LoRa-facts section removed from the system prompt
compared with the full LLM-Agent and with ADR from E4 (same seeds: paired).

Run:  python experiments/e6_ablations.py [--backend … --model …] [--plot-only]
Outputs: results/e6_summary.csv, results/e6_ablations.png
"""

from __future__ import annotations

import argparse

import numpy as np

from common import (COLORS, RESULTS, add_llm_args, append_rows, done_keys, llm_overrides,
                    load_config, mean_std, read_rows, run_one)
from netsim.config import SCENARIOS

CSV = RESULTS / "e6_summary.csv"
VARIANTS = ["LLM-Agent-noAnomalyTool", "LLM-Agent-noFacts"]


def run(scenarios, seeds, args) -> None:
    for scen in scenarios:
        cfg = load_config(scen, overrides={"agent": llm_overrides(args)})
        done = done_keys(CSV, ("scenario", "policy", "seed", "model_key"))
        for seed in seeds or cfg["seeds"]:
            for pol in VARIANTS:
                if (scen, pol, str(seed), cfg["agent"]["model"]) in done:
                    continue
                row, _ = run_one(cfg, seed, pol, "e6", scen, args.replay_only)
                row["model_key"] = cfg["agent"]["model"]
                append_rows(CSV, [row])
                print(f"  {scen:18s} {pol:24s} seed={seed} pdr_event={row['pdr_event']:.3f}", flush=True)


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_rows(CSV) + [r for r in read_rows(RESULTS / "e4_summary.csv")
                             if r["policy"] in ("ADR", "LLM-Agent")]
    pols = ["ADR", "LLM-Agent"] + VARIANTS
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(SCENARIOS))
    w = 0.8 / len(pols)
    for i, pol in enumerate(pols):
        for ax, key in zip(axes, ("pdr_event", "energy_per_delivered_event_mj")):
            m = [mean_std(r[key] for r in rows if r["scenario"] == s and r["policy"] == pol) for s in SCENARIOS]
            ax.bar(x + i * w, [a[0] for a in m], w, yerr=[a[1] for a in m], capsize=2,
                   color=COLORS[pol], label=pol)
    for ax, yl in zip(axes, ("PDR in event window", "energy / delivered packet (mJ)")):
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([s.split("_")[0].upper() for s in SCENARIOS])
        ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_yscale("log")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(pols), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS / "e6_ablations.png", dpi=150)
    plt.close(fig)
    print("  figure written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true")
    add_llm_args(ap)
    a = ap.parse_args()
    if not a.plot_only:
        run(a.scenarios, a.seeds, a)
    plot()
