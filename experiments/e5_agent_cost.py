#!/usr/bin/env python
"""E5 — Cost of the agent, computed from the decision traces of E4 (and E6):
decision latency (median, 95th percentile), latency per LLM call, LLM calls
per simulated hour, invalid-action rate, tool usage, per model.

No new simulation: every number comes from results/traces/*/*.jsonl, which
the agent writes during E4/E6.  Run E4 with another model to add it:
    python experiments/e4_anomalies.py --policies LLM-Agent --backend ollama --model qwen2.5:7b
    python experiments/e5_agent_cost.py

Outputs: results/e5_agent_cost.csv, results/e5_decisions.csv, results/e5_agent_cost.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from collections import Counter, defaultdict

import numpy as np

from common import RESULTS, TRACES, load_config

OUT = RESULTS / "e5_agent_cost.csv"
DEC = RESULTS / "e5_decisions.csv"


def collect() -> None:
    cfg = load_config()
    dec_rows, by_model = [], defaultdict(list)
    for path in sorted(TRACES.glob("*/*.jsonl")):
        exp = path.parent.name
        scen, pol, model, seed = path.stem.split("__")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            tools = Counter(m["function"]["name"] for msg in d["messages"] if msg.get("tool_calls")
                            for m in msg["tool_calls"])
            row = {"experiment": exp, "scenario": scen, "policy": pol, "model": model,
                   "seed": int(seed.replace("seed", "")), "sim_time_s": d["sim_time"],
                   "triggers": "+".join(sorted({r["trigger"] for r in d["reasons"]})),
                   "latency_s": d["latency_s"], "llm_calls": d["llm_calls"],
                   "tool_calls": d["tool_calls"], "set_calls": d["set_calls"],
                   "set_rejected": d["set_rejected"], "unknown_or_malformed": d["unknown_or_malformed_calls"],
                   "rbw_violations": d["read_before_write_violations"], "fallback": int(d["fallback"]),
                   "timeout": int(d["error"] == "timeout"), "devices_targeted": d["devices_targeted"],
                   **{f"n_{k}": v for k, v in tools.items()}}
            dec_rows.append(row)
            by_model[(model, pol, exp)].append((row, d["llm_latencies_s"]))
    if not dec_rows:
        print("  no traces yet")
        return
    keys = sorted({k for r in dec_rows for k in r}, key=lambda k: (k.startswith("n_"), k))
    with DEC.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in dec_rows:
            w.writerow({k: r.get(k, 0) for k in keys})
    out = []
    for (model, pol, exp), items in sorted(by_model.items()):
        rows = [x[0] for x in items]
        lat = np.array([r["latency_s"] for r in rows if not r["fallback"]])
        per_call = np.array([v for _, ls in items for v in ls])
        runs = {(r["experiment"], r["scenario"], r["seed"]) for r in rows}
        dur = load_config("e3_scalability")["duration_s"] if exp == "e3" else cfg["duration_s"]
        sim_hours = len(runs) * dur / 3600.0
        sets = sum(r["set_calls"] for r in rows)
        rej = sum(r["set_rejected"] for r in rows)
        tool_tot = Counter()
        for r in rows:
            for k, v in r.items():
                if k.startswith("n_"):
                    tool_tot[k[2:]] += v
        out.append({
            "model": model, "policy": pol, "experiment": exp, "runs": len(runs), "decisions": len(rows),
            "decisions_per_sim_hour": round(len(rows) / sim_hours, 2),
            "llm_calls_per_sim_hour": round(sum(r["llm_calls"] for r in rows) / sim_hours, 2),
            "tool_calls_per_decision": round(float(np.mean([r["tool_calls"] for r in rows])), 2),
            "decision_latency_median_s": round(float(np.median(lat)), 2) if len(lat) else "nan",
            "decision_latency_p95_s": round(float(np.percentile(lat, 95)), 2) if len(lat) else "nan",
            "llm_call_latency_median_s": round(float(np.median(per_call)), 2) if len(per_call) else "nan",
            "set_calls": sets, "set_rejected": rej,
            "invalid_action_rate": round(rej / sets, 3) if sets else "nan",
            "decisions_with_a_change": sum(1 for r in rows if r["set_calls"] > r["set_rejected"]),
            "unknown_or_malformed_calls": sum(r["unknown_or_malformed"] for r in rows),
            "rbw_violations": sum(r["rbw_violations"] for r in rows),
            "fallbacks": sum(r["fallback"] for r in rows), "timeouts": sum(r["timeout"] for r in rows),
            "tool_usage": json.dumps(dict(tool_tot)),
            "host": f"{platform.machine()} {os.cpu_count()} CPU threads",
        })
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("  wrote", OUT.name, DEC.name)


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with DEC.open() as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        if r["fallback"] == "0":
            groups[f"{r['model']} / {r['policy']}"].append(float(r["latency_s"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, v in sorted(groups.items()):
        v = np.sort(v)
        axes[0].plot(v, np.arange(1, len(v) + 1) / len(v), label=f"{name} (n={len(v)})")
    axes[0].set_xlabel("decision latency (s, wall clock)")
    axes[0].set_ylabel("CDF")
    axes[0].axvline(600, color="gray", ls="--", lw=1)
    axes[0].text(610, 0.1, "reporting\ninterval", fontsize=7, color="gray")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3)
    trig = defaultdict(list)
    for r in rows:
        for t in r["triggers"].split("+"):
            trig[t].append(float(r["latency_s"]))
    names = sorted(trig)
    axes[1].barh(names, [len(trig[n]) for n in names], color="#d62728")
    axes[1].set_xlabel("number of agent invocations (all runs)")
    axes[1].set_title("what triggered the agent", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "e5_agent_cost.png", dpi=150)
    plt.close(fig)
    print("  figure written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        collect()
    if DEC.exists():
        plot()
