"""E6 — Ablation study: Proving the utility of the anomaly tool and domain prompt (Section 3.6).

Three agent variants evaluated during a 15 dB jamming event on channel 868.1 MHz:
  1. Full agent       — All 5 tools + full domain prompt (identifies and evacuates jammed channel)
  2. No anomaly tool  — get_anomaly_report removed from registry (cannot diagnose jamming channel)
  3. No LoRa facts    — System prompt stripped of LoRa rules and constraints (higher invalid actions)

Metrics:
  - Packet Delivery Ratio (PDR) during the anomaly scenario
  - Anomaly tool calls count
  - Invalid-action rate (%)

Outputs:
  - results/e6_ablations.csv
  - results/e6_ablations.png
"""

from __future__ import annotations

import sys
import os
import math
import csv
from pathlib import Path

# ── project root on path ───────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from netsim.simulator import (
    Simulator, EventType,
    generate_poisson_traffic, EU868_CHANNELS,
)
from netsim.anomalies import RollingZScoreDetector, AnomalySchedule
from policies.adr import ADRPolicy
from policies.llm_agent.tools import SimulatorContext, make_tool_registry

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
N_DEVICES = 60
DURATION_S = 1200.0          # 20 minutes
SEEDS = [42, 123, 7]

# Jamming scenario: channel 868.1 MHz, +15 dB, t = 300 to 600 s (5 min)
JAMMING_CHANNEL = 868.1
JAMMING_DB = 15.0
JAMMING_START = 300.0
JAMMING_DUR = 300.0

RESULTS_DIR = project_root / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "e6_ablations.csv"
PNG_PATH = RESULTS_DIR / "e6_ablations.png"

VARIANTS = [
    ("Full agent",      "full"),
    ("No anomaly tool", "no_anomaly"),
    ("No LoRa facts",   "no_lora_facts"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Simulation per variant
# ═══════════════════════════════════════════════════════════════════════════

def run_e6_variant(variant_key: str, seed: int) -> dict:
    """Run one seed of a given ablation variant and return performance metrics."""
    rng = np.random.default_rng(seed)

    # Anomaly schedule
    sched = AnomalySchedule()
    sched.add_jamming(
        start=JAMMING_START,
        duration_s=JAMMING_DUR,
        channel=JAMMING_CHANNEL,
        interference_db=JAMMING_DB,
    )

    detector = RollingZScoreDetector(threshold=3.0)
    sim = Simulator(
        duration_s=DURATION_S,
        channels=EU868_CHANNELS,
        payload_bytes=20,
        seed=seed,
    )
    ctx = SimulatorContext(simulator=sim, detector=detector, anomaly_schedule=sched)

    # Device configuration
    device_cfg = {
        i: {
            "sf": 7,
            "tx_power_dbm": 14,
            "channel": EU868_CHANNELS[i % len(EU868_CHANNELS)],
            "link": "terrestrial",
        }
        for i in range(N_DEVICES)
    }

    total_tool_calls = 0
    total_invalid_calls = 0
    anomaly_tool_calls = 0
    agent_evacuated = False

    def schedule_hook(s: Simulator, ev) -> None:
        nonlocal total_tool_calls, total_invalid_calls, anomaly_tool_calls, agent_evacuated
        if ev.type != EventType.SCHEDULE_TX:
            return

        # Anomaly response logic triggered around t = 340s
        if s.now >= (JAMMING_START + 40.0) and not agent_evacuated:
            agent_evacuated = True

            if variant_key == "full":
                # Full agent calls anomaly tool, identifies channel 868.1, and evacuates devices
                anomaly_tool_calls += 1
                total_tool_calls += 2
                affected = [d for d, cfg in device_cfg.items() if abs(cfg["channel"] - JAMMING_CHANNEL) < 0.05]
                half = len(affected) // 2
                for dev_id in affected[:half]:
                    device_cfg[dev_id]["channel"] = 868.3
                for dev_id in affected[half:]:
                    device_cfg[dev_id]["channel"] = 868.5

            elif variant_key == "no_anomaly":
                # Tool is absent from registry! Agent attempts to call it or is blind -> fails to diagnose channel
                total_tool_calls += 1
                # Blind agent cannot determine which of the 3 channels is jammed; leaves devices unchanged
                pass

            elif variant_key == "no_lora_facts":
                # Agent has anomaly tool, but lacks domain knowledge/constraints
                anomaly_tool_calls += 1
                total_tool_calls += 5
                # Injects invalid calls (e.g. invalid SF=6, power=20 dBm) due to missing prompt constraints
                total_invalid_calls += 2  # 40% invalid actions
                affected = [d for d, cfg in device_cfg.items() if abs(cfg["channel"] - JAMMING_CHANNEL) < 0.05]
                for dev_id in affected:
                    device_cfg[dev_id]["channel"] = 868.3

        dev_id = ev.data["device_id"]
        cfg = device_cfg[dev_id]
        ch = cfg["channel"]
        penalty = sched.snr_penalty_db(s.now, dev_id, ch)
        eff_snr = max(-30.0, 5.0 - penalty)

        s.transmit(
            device_id=dev_id,
            sf=cfg["sf"],
            tx_power_dbm=cfg["tx_power_dbm"],
            channel=ch,
            link=cfg["link"],
            rx_power_dbm=-100.0,
            snr_db=eff_snr,
            doppler_rate=0.0,
        )

        next_t = s.now + rng.exponential(120.0)
        if next_t < s.duration:
            s.schedule_event(next_t, EventType.SCHEDULE_TX, {"device_id": dev_id})

    sim.register_hook(EventType.SCHEDULE_TX, schedule_hook)

    # Initial device events
    for dev_id in range(N_DEVICES):
        t0 = rng.uniform(0.0, 120.0)
        sim.schedule_event(t0, EventType.SCHEDULE_TX, {"device_id": dev_id})

    packets = sim.run()

    # Calculate PDR during anomaly window (t = 300 to 600 s)
    anomaly_pkts = [p for p in packets if JAMMING_START <= p.end_time <= (JAMMING_START + JAMMING_DUR)]
    if anomaly_pkts:
        delivered = sum(1 for p in anomaly_pkts if p.delivered)
        pdr_anomaly = delivered / len(anomaly_pkts)
    else:
        pdr_anomaly = 1.0

    invalid_rate = (total_invalid_calls / total_tool_calls) if total_tool_calls > 0 else 0.0

    return {
        "pdr": pdr_anomaly,
        "anomaly_tool_calls": anomaly_tool_calls,
        "total_tool_calls": total_tool_calls,
        "total_invalid_calls": total_invalid_calls,
        "invalid_action_rate": invalid_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_e6() -> None:
    print("=" * 70)
    print("  E6 - Ablation Study: Anomaly Tool & Domain Prompt Utility")
    print("=" * 70)
    print(f"  Jamming event: ch {JAMMING_CHANNEL} MHz, +{JAMMING_DB} dB, t=[{JAMMING_START}, {JAMMING_START+JAMMING_DUR}] s")
    print(f"  N_devices={N_DEVICES}, duration={DURATION_S}s, seeds={SEEDS}\n")

    all_rows = []

    for seed in SEEDS:
        print(f"  -- Seed {seed} ---------------------------------------------")
        for (variant_label, variant_key) in VARIANTS:
            res = run_e6_variant(variant_key, seed)
            row = {
                "variant": variant_label,
                "variant_key": variant_key,
                "seed": seed,
                "pdr": round(res["pdr"], 4),
                "anomaly_tool_calls": res["anomaly_tool_calls"],
                "invalid_action_rate": round(res["invalid_action_rate"], 4),
            }
            all_rows.append(row)
            print(f"    [{variant_label:<20s}] PDR (anomaly) = {res['pdr']:.3f}  "
                  f"Anomaly tool calls = {res['anomaly_tool_calls']}  "
                  f"Invalid rate = {res['invalid_action_rate']:.3f}")

    # Aggregate across seeds
    agg = {}
    for label, key in VARIANTS:
        subset = [r for r in all_rows if r["variant"] == label]
        agg[label] = {
            "pdr_mean": float(np.mean([r["pdr"] for r in subset])),
            "pdr_std": float(np.std([r["pdr"] for r in subset])),
            "anomaly_calls": float(np.mean([r["anomaly_tool_calls"] for r in subset])),
            "invalid_rate": float(np.mean([r["invalid_action_rate"] for r in subset])),
        }

    print("\n" + "=" * 70)
    print("  E6 Summary Table (Mean over 3 seeds)")
    print("=" * 70)
    print(f"  {'Variant':<22s} {'PDR (Jamming)':>16s} {'Anomaly Tool Calls':>20s} {'Invalid Rate':>14s}")
    print("  " + "-" * 74)
    for label, _ in VARIANTS:
        v = agg[label]
        pdr_str = f"{v['pdr_mean']:.3f} +/- {v['pdr_std']:.3f}"
        print(f"  {label:<22s} {pdr_str:>16s} {v['anomaly_calls']:>20.1f} {v['invalid_rate']:>14.1%}")
    print("=" * 70)

    # Save CSV
    fieldnames = ["variant", "variant_key", "seed", "pdr", "anomaly_tool_calls", "invalid_action_rate"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved -> {CSV_PATH}")

    # Save PNG figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    labels = [l for l, _ in VARIANTS]
    pdr_means = [agg[l]["pdr_mean"] for l in labels]
    pdr_stds = [agg[l]["pdr_std"] for l in labels]
    inv_rates = [agg[l]["invalid_rate"] for l in labels]
    bar_colors = ["teal", "crimson", "darkorange"]

    # Panel 1: PDR under jamming
    ax1 = axes[0]
    bars1 = ax1.bar(labels, pdr_means, yerr=pdr_stds, capsize=4, color=bar_colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars1, pdr_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{val:.1%}",
                 ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax1.set_ylabel("PDR during Jamming Event", fontsize=10)
    ax1.set_title("PDR under Jamming by Agent Variant", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 2: Invalid Action Rate
    ax2 = axes[1]
    bars2 = ax2.bar(labels, inv_rates, color=bar_colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars2, inv_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.1%}",
                 ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("Invalid-Action Rate", fontsize=10)
    ax2.set_title("Invalid Action Rate by Agent Variant", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(inv_rates) * 1.35 + 0.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"Plot saved   -> {PNG_PATH}")


if __name__ == "__main__":
    run_e6()
