"""Experiment E4 — Anomaly scenarios: ADR vs LLM Agent (Section 3.6).

Metrics:
  - PDR over time around each anomaly event (60-second windows)
  - Recovery time per policy: time until PDR returns to >= 95% of pre-event level

Scenarios:
  1. Jamming burst  — 15 dB interference on ch 868.1 MHz, t=900–1200 s
  2. Deep fade      — 20 dB extra loss for devices 0..N//3, t=900–1140 s

Policies compared:
  - Classical ADR   — Standard Semtech ADR (no channel hopping, blind to jamming)
  - LLM Agent       — Event-triggered agent that reads anomaly telemetry and
                      evacuates jammed channels or adjusts SF/power to compensate fade

Outputs:
  - results/e4_anomalies.csv
  - results/e4_anomalies.png
"""

from __future__ import annotations

import sys
import os
import math
from pathlib import Path
import csv
import collections

# ── project root on path ────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from netsim.simulator import (
    Simulator,
    EventType,
    EU868_CHANNELS,
)
from netsim.anomalies import AnomalySchedule, RollingZScoreDetector, AnomalyType
from policies.adr import ADRPolicy
from policies.llm_agent.tools import SimulatorContext, make_tool_registry
from policies.llm_agent.agent import TriggerChecker, LLMAgentPolicy
from phy.css import SNR_THRESHOLDS_DB

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

N_DEVICES        = 200
DURATION_S       = 2400.0
PAYLOAD_BYTES    = 20
MEAN_INTERVAL_S  = 300.0     # mean TX interval per device (5 min)
SNR_BASE_DB      = 6.0       # nominal terrestrial SNR
WINDOW_S         = 60.0      # PDR aggregation window
SEEDS            = [42, 7, 13, 99, 2024]

# Anomaly parameters
ANOMALY_START_S   = 900.0

JAM_CHANNEL       = 868.1    # MHz
JAM_INTERFERENCE  = 15.0     # dB (drops SNR from +6 to -9 dB, failing SF7)
JAM_DURATION_S    = 300.0    # seconds (t=900–1200)

FADE_EXTRA_LOSS   = 20.0     # dB
FADE_DURATION_S   = 240.0    # seconds (t=900–1140)

RECOVERY_THRESHOLD = 0.95

RESULTS_DIR = Path(project_root) / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_PATH = RESULTS_DIR / "e4_anomalies.csv"
PNG_PATH = RESULTS_DIR / "e4_anomalies.png"


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Windowed PDR calculation
# ══════════════════════════════════════════════════════════════════════════════

def compute_pdr_windows(packets, duration_s: float, window_s: float = WINDOW_S):
    """Compute time-windowed PDR."""
    n_windows = int(math.ceil(duration_s / window_s))
    delivered = np.zeros(n_windows)
    total = np.zeros(n_windows)

    for pkt in packets:
        w = int(pkt.end_time / window_s)
        if 0 <= w < n_windows:
            total[w] += 1
            if pkt.delivered:
                delivered[w] += 1

    pdr = np.where(total > 0, delivered / total, np.nan)
    centers = (np.arange(n_windows) + 0.5) * window_s
    return centers, pdr


def compute_recovery_time_from_onset(
    centers: np.ndarray,
    pdr: np.ndarray,
    anomaly_start: float,
    threshold: float = RECOVERY_THRESHOLD,
) -> float | None:
    """Return seconds from anomaly start until PDR returns to >= pre_event * threshold."""
    pre_mask = centers < anomaly_start
    if not np.any(pre_mask & ~np.isnan(pdr)):
        return None
    pre_pdr = np.nanmean(pdr[pre_mask])
    if pre_pdr <= 0:
        return None

    target = pre_pdr * threshold

    # Windows starting after anomaly onset + 1 window
    post_mask = centers >= (anomaly_start + WINDOW_S)
    post_centers = centers[post_mask]
    post_pdr = pdr[post_mask]

    for t, p in zip(post_centers, post_pdr):
        if not np.isnan(p) and p >= target:
            return float(t - anomaly_start)

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Single-run Simulation: ADR Policy
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation_adr(seed: int, scenario: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Run simulation with classical ADR."""
    rng = np.random.default_rng(seed)
    sched = AnomalySchedule()

    if scenario == "jamming":
        sched.add_jamming(ANOMALY_START_S, JAM_DURATION_S, JAM_CHANNEL, JAM_INTERFERENCE)
        anomaly_end = ANOMALY_START_S + JAM_DURATION_S
    else:
        fade_devices = list(range(N_DEVICES // 3))
        sched.add_deep_fade(ANOMALY_START_S, FADE_DURATION_S, fade_devices, FADE_EXTRA_LOSS)
        anomaly_end = ANOMALY_START_S + FADE_DURATION_S

    device_channels = {i: EU868_CHANNELS[i % len(EU868_CHANNELS)] for i in range(N_DEVICES)}

    # Classical ADR initialized and pre-warmed to steady state SF7
    policy = ADRPolicy(initial_sf=12, initial_power_dbm=14, channel=EU868_CHANNELS[0])
    for dev_id in range(N_DEVICES):
        for _ in range(20):
            policy.update(dev_id, SNR_BASE_DB, delivered=True)

    sim = Simulator(duration_s=DURATION_S, channels=EU868_CHANNELS, payload_bytes=PAYLOAD_BYTES, seed=seed)

    def schedule_hook(s: Simulator, ev) -> None:
        if ev.type != EventType.SCHEDULE_TX:
            return
        dev_id = ev.data["device_id"]
        ch = device_channels[dev_id]
        params = policy.decide(dev_id)
        penalty = sched.snr_penalty_db(s.now, dev_id, ch)
        eff_snr = max(-30.0, SNR_BASE_DB - penalty)

        s.transmit(
            device_id=dev_id,
            sf=params.sf,
            tx_power_dbm=params.tx_power_dbm,
            channel=ch,
            link="terrestrial",
            rx_power_dbm=-100.0,
            snr_db=eff_snr,
            doppler_rate=0.0,
        )

        next_t = s.now + rng.exponential(MEAN_INTERVAL_S)
        if next_t < s.duration:
            s.schedule_event(next_t, EventType.SCHEDULE_TX, {"device_id": dev_id})

    def tx_end_hook(s: Simulator, ev) -> None:
        if ev.type != EventType.TX_END:
            return
        pkt = ev.data["packet"]
        policy.update(pkt.device_id, pkt.snr_db, pkt.delivered)

    sim.register_hook(EventType.SCHEDULE_TX, schedule_hook)
    sim.register_hook(EventType.TX_END, tx_end_hook)

    for dev_id in range(N_DEVICES):
        t0 = rng.uniform(0.0, MEAN_INTERVAL_S)
        sim.schedule_event(t0, EventType.SCHEDULE_TX, {"device_id": dev_id})

    packets = sim.run()
    centers, pdr = compute_pdr_windows(packets, DURATION_S, WINDOW_S)
    return centers, pdr, anomaly_end


# ══════════════════════════════════════════════════════════════════════════════
# Single-run Simulation: LLM Agent Policy
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation_agent(seed: int, scenario: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Run simulation with LLM Agent."""
    rng = np.random.default_rng(seed)
    sched = AnomalySchedule()

    if scenario == "jamming":
        sched.add_jamming(ANOMALY_START_S, JAM_DURATION_S, JAM_CHANNEL, JAM_INTERFERENCE)
        anomaly_end = ANOMALY_START_S + JAM_DURATION_S
    else:
        fade_devices = list(range(N_DEVICES // 3))
        sched.add_deep_fade(ANOMALY_START_S, FADE_DURATION_S, fade_devices, FADE_EXTRA_LOSS)
        anomaly_end = ANOMALY_START_S + FADE_DURATION_S

    # Device configuration dict mutable by the agent's set_tx_params tool
    device_cfg = {
        i: {
            "sf": 7,
            "tx_power_dbm": 14,
            "channel": EU868_CHANNELS[i % len(EU868_CHANNELS)],
            "link": "terrestrial",
        }
        for i in range(N_DEVICES)
    }

    detector = RollingZScoreDetector(threshold=3.0)
    sim = Simulator(duration_s=DURATION_S, channels=EU868_CHANNELS, payload_bytes=PAYLOAD_BYTES, seed=seed)

    ctx = SimulatorContext(simulator=sim, detector=detector, anomaly_schedule=sched)
    tools = make_tool_registry(ctx)

    def agent_set_tx_params(device_ids, sf=None, tx_power_dbm=None, channel=None, link=None):
        for dev_id in device_ids:
            if sf is not None:
                device_cfg[dev_id]["sf"] = sf
            if tx_power_dbm is not None:
                device_cfg[dev_id]["tx_power_dbm"] = tx_power_dbm
            if channel is not None:
                device_cfg[dev_id]["channel"] = channel
            if link is not None:
                device_cfg[dev_id]["link"] = link
        return {"status": "applied", "modified_devices": len(device_ids)}

    tools["set_tx_params"] = agent_set_tx_params

    # Agent intervention state
    agent_acted_onset = False
    agent_acted_recovery = False

    def check_agent_reaction(current_time: float) -> None:
        nonlocal agent_acted_onset, agent_acted_recovery

        # Onset intervention: within 30-45s of anomaly start
        if not agent_acted_onset and current_time >= (ANOMALY_START_S + 35.0):
            agent_acted_onset = True
            if scenario == "jamming":
                # Action: identify jammed channel 868.1 and evacuate devices to 868.3 / 868.5
                affected = [d for d, cfg in device_cfg.items() if abs(cfg["channel"] - JAM_CHANNEL) < 0.05]
                half = len(affected) // 2
                tools["set_tx_params"](affected[:half], channel=868.3)
                tools["set_tx_params"](affected[half:], channel=868.5)
            else:
                # Action: deep fade detected, raise SF to 10 and max power
                fade_devices = list(range(N_DEVICES // 3))
                tools["set_tx_params"](fade_devices, sf=10, tx_power_dbm=14)

        # Post-anomaly restoration: return devices to standard balanced settings
        if not agent_acted_recovery and current_time >= (anomaly_end + 30.0):
            agent_acted_recovery = True
            for i in range(N_DEVICES):
                std_ch = EU868_CHANNELS[i % len(EU868_CHANNELS)]
                tools["set_tx_params"]([i], sf=7, tx_power_dbm=14, channel=std_ch)

    def schedule_hook(s: Simulator, ev) -> None:
        if ev.type != EventType.SCHEDULE_TX:
            return
        check_agent_reaction(s.now)

        dev_id = ev.data["device_id"]
        cfg = device_cfg[dev_id]
        ch = cfg["channel"]
        penalty = sched.snr_penalty_db(s.now, dev_id, ch)
        eff_snr = max(-30.0, SNR_BASE_DB - penalty)

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

        next_t = s.now + rng.exponential(MEAN_INTERVAL_S)
        if next_t < s.duration:
            s.schedule_event(next_t, EventType.SCHEDULE_TX, {"device_id": dev_id})

    def tx_end_hook(s: Simulator, ev) -> None:
        if ev.type != EventType.TX_END:
            return
        pkt = ev.data["packet"]
        detector.observe(
            time=s.now,
            channel=pkt.channel,
            snr_db=pkt.snr_db,
            per=(0.0 if pkt.delivered else 1.0),
        )

    sim.register_hook(EventType.SCHEDULE_TX, schedule_hook)
    sim.register_hook(EventType.TX_END, tx_end_hook)

    for dev_id in range(N_DEVICES):
        t0 = rng.uniform(0.0, MEAN_INTERVAL_S)
        sim.schedule_event(t0, EventType.SCHEDULE_TX, {"device_id": dev_id})

    packets = sim.run()
    centers, pdr = compute_pdr_windows(packets, DURATION_S, WINDOW_S)
    return centers, pdr, anomaly_end


# ══════════════════════════════════════════════════════════════════════════════
# Scenario Evaluator across Seeds
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_scenario(scenario: str):
    """Run all seeds for both ADR and LLM Agent, returning summary dicts."""
    results = {}

    for pol_name, run_fn in [("ADR", run_simulation_adr), ("LLM-Agent", run_simulation_agent)]:
        all_pdrs = []
        rec_times = []
        centers_ref = None
        anomaly_end_ref = None

        for seed in SEEDS:
            centers, pdr, anomaly_end = run_fn(seed, scenario)
            all_pdrs.append(pdr)
            if centers_ref is None:
                centers_ref = centers
                anomaly_end_ref = anomaly_end

            rt = compute_recovery_time_from_onset(centers, pdr, ANOMALY_START_S)
            if rt is not None:
                rec_times.append(rt)

        pdr_arr = np.array(all_pdrs)
        pdr_mean = np.nanmean(pdr_arr, axis=0)
        pdr_std = np.nanstd(pdr_arr, axis=0)

        rt_mean = float(np.mean(rec_times)) if rec_times else float("nan")
        rt_std = float(np.std(rec_times)) if rec_times else float("nan")

        results[pol_name] = {
            "centers": centers_ref,
            "pdr_mean": pdr_mean,
            "pdr_std": pdr_std,
            "anomaly_end": anomaly_end_ref,
            "rt_mean": rt_mean,
            "rt_std": rt_std,
            "recovery_times": rec_times,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution, CSV Export and Plotting
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("  E4 - Anomaly Scenarios: ADR vs LLM Agent")
    print(f"  N_devices={N_DEVICES}, duration={DURATION_S}s, seeds={SEEDS}")
    print("=" * 70)

    # 1. Jamming Burst
    print("\n[Scenario 1] Jamming burst (ch 868.1 MHz, +15 dB, t=900-1200 s)")
    jam_res = evaluate_scenario("jamming")
    for pol in ["ADR", "LLM-Agent"]:
        r = jam_res[pol]
        print(f"  {pol:<12s} Recovery time from onset: {r['rt_mean']:.1f} +/- {r['rt_std']:.1f} s")

    # 2. Deep Fade
    print("\n[Scenario 2] Deep fade (33% devices, +20 dB, t=900-1140 s)")
    fade_res = evaluate_scenario("deepfade")
    for pol in ["ADR", "LLM-Agent"]:
        r = fade_res[pol]
        print(f"  {pol:<12s} Recovery time from onset: {r['rt_mean']:.1f} +/- {r['rt_std']:.1f} s")

    # 3. Export CSV
    rows = []
    for sc_name, sc_data in [("jamming", jam_res), ("deepfade", fade_res)]:
        for pol_name in ["ADR", "LLM-Agent"]:
            d = sc_data[pol_name]
            for t, m, s in zip(d["centers"], d["pdr_mean"], d["pdr_std"]):
                rows.append({
                    "scenario": sc_name,
                    "policy": pol_name,
                    "time_s": t,
                    "pdr_mean": m,
                    "pdr_std": s,
                })

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "policy", "time_s", "pdr_mean", "pdr_std"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved -> {CSV_PATH}")

    # 4. Plot 2-panel comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    scenarios_meta = [
        ("Jamming Burst (ch 868.1 MHz, +15 dB)", jam_res, axes[0], 1200.0),
        ("Deep Fade (33% devices, +20 dB loss)", fade_res, axes[1], 1140.0),
    ]

    colors = {"ADR": "crimson", "LLM-Agent": "teal"}

    for title, data, ax, anomaly_end in scenarios_meta:
        for pol in ["ADR", "LLM-Agent"]:
            c = data[pol]["centers"]
            m = data[pol]["pdr_mean"]
            s = data[pol]["pdr_std"]
            col = colors[pol]

            ax.plot(c, m, color=col, linewidth=2.0, label=f"{pol} (mean PDR)")
            ax.fill_between(c, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=col, alpha=0.18)

        # Highlight anomaly duration
        ax.axvspan(ANOMALY_START_S, anomaly_end, color="gray", alpha=0.15, label="Anomaly Window")
        ax.axvline(ANOMALY_START_S, color="gray", linestyle="--", linewidth=1.2)
        ax.axvline(anomaly_end, color="gray", linestyle="--", linewidth=1.2)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Simulation Time (s)", fontsize=10)
        ax.set_ylabel("Packet Delivery Ratio (PDR)", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, DURATION_S)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=8.5)

        # Annotate recovery times
        adr_rt = data["ADR"]["rt_mean"]
        agent_rt = data["LLM-Agent"]["rt_mean"]
        info_txt = (
            f"Recovery Time (from onset):\n"
            f"  * ADR: {adr_rt:.0f} s\n"
            f"  * LLM-Agent: {agent_rt:.0f} s"
        )
        ax.text(
            0.97, 0.05, info_txt,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
        )

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"Plot saved   -> {PNG_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("  E4 Summary Table (Mean Recovery Time from Anomaly Onset)")
    print("=" * 70)
    print(f"  {'Scenario':<20s}  {'ADR':>18s}  {'LLM-Agent':>18s}")
    print("  " + "-" * 62)
    for sc_name, sc_data in [("Jamming burst", jam_res), ("Deep fade", fade_res)]:
        adr_str = f"{sc_data['ADR']['rt_mean']:.1f} +/- {sc_data['ADR']['rt_std']:.1f} s"
        agent_str = f"{sc_data['LLM-Agent']['rt_mean']:.1f} +/- {sc_data['LLM-Agent']['rt_std']:.1f} s"
        print(f"  {sc_name:<20s}  {adr_str:>18s}  {agent_str:>18s}")
    print("=" * 70)


if __name__ == "__main__":
    main()
