"""Experiment E3 — Scalability sweep, no anomalies (Section 3.6).

Sweeps N_devices ∈ [100, 200, 300, 500, 700, 1000] over 5 seeds and
compares three policies (Fixed-SF12, Fixed-SF7, ADR) on:
  - PDR  (packet delivery ratio)
  - Energy per delivered packet (mJ)

Outputs:
  results/e3_scalability.csv   — raw per-run data
  results/e3_scalability.png   — 2-panel figure (PDR & energy vs N)
"""

from __future__ import annotations

import sys
import os

# Ensure project root is importable regardless of working directory
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

import csv
import pathlib
import collections

import numpy as np

matplotlib_available = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_available = False

from netsim.simulator import (
    Simulator,
    EventType,
    generate_poisson_traffic,
    _default_schedule_hook,
)
from policies.fixed import FixedSF12MaxPower, FixedSF7
from policies.adr import ADRPolicy
from phy.css import SNR_THRESHOLDS_DB

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SEED_GLOBAL = 42

N_DEVICES_LIST = [100, 200, 300, 500, 700, 1000]
DURATION_S = 3600.0
SEEDS = [0, 1, 2, 3, 4]
PAYLOAD_BYTES = 20
MEAN_INTERVAL_S = 600.0
TX_POWER_DBM = 14
RX_POWER_DBM = -80.0
LINK = "terrestrial"
EU868_CHANNELS = [868.1, 868.3, 868.5]

# Channel SNR (dB) for nominal terrestrial link (rural cell, devices <= 3 km)
SNR_NOMINAL = 5.0   # link closes easily for all SFs (SF7 requires -7.5 dB)

RESULTS_DIR = pathlib.Path(_PROJECT_ROOT) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "e3_scalability.csv"
PNG_PATH = RESULTS_DIR / "e3_scalability.png"


# ═══════════════════════════════════════════════════════════════════════════
# PER lookup
# ═══════════════════════════════════════════════════════════════════════════

def simple_per(sf: int, snr_db: float, doppler_rate: float = 0.0) -> float:
    """Zero PER above threshold, unit PER below."""
    return 0.0 if snr_db >= SNR_THRESHOLDS_DB[sf] else 1.0


# ═══════════════════════════════════════════════════════════════════════════
# ADR hook factory
# ═══════════════════════════════════════════════════════════════════════════

def make_adr_hooks(policy: ADRPolicy, snr_db: float):
    """Return SCHEDULE_TX and TX_END hooks for closed-loop ADR adaptation."""
    def schedule_hook(sim_obj: Simulator, event) -> None:
        if event.type != EventType.SCHEDULE_TX:
            return
        d = event.data
        tx_params = policy.decide(d["device_id"])
        sim_obj.transmit(
            device_id=d["device_id"],
            sf=tx_params.sf,
            tx_power_dbm=tx_params.tx_power_dbm,
            channel=d["channel"],
            link=d["link"],
            rx_power_dbm=d["rx_power_dbm"],
            snr_db=snr_db,
            doppler_rate=0.0,
        )

    def tx_end_hook(sim_obj: Simulator, event) -> None:
        if event.type != EventType.TX_END:
            return
        pkt = event.data["packet"]
        policy.update(pkt.device_id, pkt.snr_db, pkt.delivered)

    return schedule_hook, tx_end_hook


# ═══════════════════════════════════════════════════════════════════════════
# Single-run helpers
# ═══════════════════════════════════════════════════════════════════════════

def run_fixed(sf: int, snr_db: float, n_devices: int, seed: int) -> dict:
    """One simulation run for a fixed-SF policy."""
    sim = Simulator(
        duration_s=DURATION_S,
        channels=EU868_CHANNELS,
        payload_bytes=PAYLOAD_BYTES,
        seed=seed,
        per_lookup=simple_per,
    )
    sim.register_hook(EventType.SCHEDULE_TX, _default_schedule_hook)
    generate_poisson_traffic(
        sim,
        n_devices=n_devices,
        mean_interval_s=MEAN_INTERVAL_S,
        sf=sf,
        tx_power_dbm=TX_POWER_DBM,
        link=LINK,
        snr_db=snr_db,
        rx_power_dbm=RX_POWER_DBM,
        doppler_rate=0.0,
    )
    sim.run()
    return sim.metrics()


def run_adr(n_devices: int, seed: int, snr_db: float = SNR_NOMINAL) -> dict:
    """One simulation run for the ADR policy."""
    policy = ADRPolicy(
        initial_sf=12,
        initial_power_dbm=TX_POWER_DBM,
        channel=EU868_CHANNELS[0],
        link=LINK,
    )

    # Pre-warm policy with nominal history so devices adapt from initial SF12
    for dev_id in range(n_devices):
        for _ in range(20):
            policy.update(dev_id, snr_db, delivered=True)

    sim = Simulator(
        duration_s=DURATION_S,
        channels=EU868_CHANNELS,
        payload_bytes=PAYLOAD_BYTES,
        seed=seed,
        per_lookup=simple_per,
    )

    # Register ADR hooks for schedule and dynamic live feedback
    schedule_hook, tx_end_hook = make_adr_hooks(policy, snr_db)
    sim.register_hook(EventType.SCHEDULE_TX, schedule_hook)
    sim.register_hook(EventType.TX_END, tx_end_hook)

    # Pre-schedule Poisson traffic (SF/power overridden by ADR hook)
    generate_poisson_traffic(
        sim,
        n_devices=n_devices,
        mean_interval_s=MEAN_INTERVAL_S,
        sf=12,
        tx_power_dbm=TX_POWER_DBM,
        link=LINK,
        snr_db=snr_db,
        rx_power_dbm=RX_POWER_DBM,
        doppler_rate=0.0,
    )

    sim.run()
    return sim.metrics()


# ═══════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    policies = [
        ("Fixed-SF12", lambda n, s: run_fixed(12, SNR_NOMINAL, n, s)),
        ("Fixed-SF7",  lambda n, s: run_fixed(7,  SNR_NOMINAL, n, s)),
        ("ADR",        lambda n, s: run_adr(n, s, SNR_NOMINAL)),
    ]

    rows: list[dict] = []

    total_runs = len(policies) * len(N_DEVICES_LIST) * len(SEEDS)
    run_idx = 0

    for policy_name, run_fn in policies:
        for n_dev in N_DEVICES_LIST:
            for seed in SEEDS:
                run_idx += 1
                print(
                    f"[{run_idx:3d}/{total_runs}]  "
                    f"policy={policy_name:<12s}  n_devices={n_dev:5d}  seed={seed}",
                    flush=True,
                )
                m = run_fn(n_dev, seed)
                rows.append({
                    "policy":                  policy_name,
                    "n_devices":               n_dev,
                    "seed":                    seed,
                    "pdr":                     m["pdr"],
                    "energy_mj_per_delivered": m["energy_per_delivered_mj"],
                })

    # ── Write CSV ──────────────────────────────────────────────────────────
    fieldnames = ["policy", "n_devices", "seed", "pdr", "energy_mj_per_delivered"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved -> {CSV_PATH}")

    # ── Aggregate stats ───────────────────────────────────────────────────
    # (policy, n_devices) → list of values
    pdr_data: dict[tuple, list[float]] = collections.defaultdict(list)
    energy_data: dict[tuple, list[float]] = collections.defaultdict(list)

    for r in rows:
        key = (r["policy"], r["n_devices"])
        pdr_data[key].append(r["pdr"])
        energy_data[key].append(r["energy_mj_per_delivered"])

    # Print summary table
    print()
    print(
        f"{'Policy':<14} {'N_dev':>6}  {'PDR mean':>10} {'PDR std':>9}  "
        f"{'E/pkt(mJ) mean':>15} {'E/pkt(mJ) std':>14}"
    )
    print("-" * 75)
    for policy_name, _ in policies:
        for n_dev in N_DEVICES_LIST:
            key = (policy_name, n_dev)
            pdrs = pdr_data[key]
            engs = energy_data[key]
            print(
                f"{policy_name:<14} {n_dev:>6d}  "
                f"{np.mean(pdrs):>10.4f} {np.std(pdrs):>9.4f}  "
                f"{np.mean(engs):>15.3f} {np.std(engs):>14.3f}"
            )

    # ── Plot ───────────────────────────────────────────────────────────────
    if not matplotlib_available:
        print("matplotlib not available — skipping plot.")
        return

    colors = {"Fixed-SF12": "tab:blue", "Fixed-SF7": "tab:orange", "ADR": "tab:green"}
    markers = {"Fixed-SF12": "o", "Fixed-SF7": "s", "ADR": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for policy_name, _ in policies:
        x = N_DEVICES_LIST
        pdr_means = [np.mean(pdr_data[(policy_name, n)]) for n in x]
        pdr_stds  = [np.std(pdr_data[(policy_name, n)])  for n in x]
        eng_means = [np.mean(energy_data[(policy_name, n)]) for n in x]
        eng_stds  = [np.std(energy_data[(policy_name, n)])  for n in x]

        col = colors[policy_name]
        mrk = markers[policy_name]

        # PDR panel
        ax = axes[0]
        ax.plot(x, pdr_means, marker=mrk, color=col, label=policy_name)
        ax.fill_between(
            x,
            [m - s for m, s in zip(pdr_means, pdr_stds)],
            [m + s for m, s in zip(pdr_means, pdr_stds)],
            alpha=0.2, color=col,
        )

        # Energy panel
        ax = axes[1]
        ax.plot(x, eng_means, marker=mrk, color=col, label=policy_name)
        ax.fill_between(
            x,
            [m - s for m, s in zip(eng_means, eng_stds)],
            [m + s for m, s in zip(eng_means, eng_stds)],
            alpha=0.2, color=col,
        )

    # Formatting
    axes[0].set_xlabel("Number of Devices")
    axes[0].set_ylabel("PDR (Packet Delivery Ratio)")
    axes[0].set_title("E3 Scalability — PDR vs N_devices")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_xlabel("Number of Devices")
    axes[1].set_ylabel("Energy per Delivered Packet (mJ)")
    axes[1].set_title("E3 Scalability — Energy vs N_devices")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"Figure saved  -> {PNG_PATH}")


if __name__ == "__main__":
    main()
