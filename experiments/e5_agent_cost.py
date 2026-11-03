"""E5 — Agent cost measurement (Section 3.6).

Measures:
  - Wall-clock latency per decision: median and P95 (seconds)
  - LLM calls per simulated hour (based on trigger rate)
  - Invalid-action rate: rejected set_tx_params / total set_tx_params calls
  - Tool calls per decision: mean

Uses a MockLLMAgentPolicy that overrides _run_agent to avoid requiring
a live ollama server.

Outputs:
  - results/e5_agent_cost.csv
  - results/e5_agent_cost.png
"""

import sys
import os
import time as wallclock
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
    Simulator, Packet, EventType,
    generate_poisson_traffic, _default_schedule_hook, EU868_CHANNELS,
)
from netsim.anomalies import RollingZScoreDetector, AnomalySchedule
from policies.adr import ADRPolicy
from policies.llm_agent.agent import LLMAgentPolicy, AgentDecision, TriggerChecker
from policies.llm_agent.tools import SimulatorContext, make_tool_registry
from policies.llm_agent.tools import (
    VALID_SFS, VALID_CHANNELS, VALID_LINKS, VALID_POWER_RANGE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
N_DECISIONS = 100          # mock agent decisions to simulate
N_DEVICES = 100
DURATION_S = 7200.0        # 2 hours
RESULTS_DIR = project_root / "results"


# ═══════════════════════════════════════════════════════════════════════════
# Mock LLM Agent Policy
# ═══════════════════════════════════════════════════════════════════════════

class MockLLMAgentPolicy(LLMAgentPolicy):
    """LLMAgentPolicy with _run_agent overridden to avoid real ollama calls.

    Simulates realistic agent behaviour:
      - Sleeps a random amount to emulate LLM latency.
      - Calls 1-3 random read tools.
      - Calls set_tx_params with valid args ~80 % of the time,
        invalid args ~20 % of the time.
    """

    def __init__(self, ctx: SimulatorContext, rng: np.random.Generator,
                 adr: ADRPolicy | None = None,
                 trigger_checker: TriggerChecker | None = None,
                 tool_registry_override: dict | None = None) -> None:
        super().__init__(ctx=ctx, adr=adr, trigger_checker=trigger_checker)
        self._mock_rng = rng
        # Allow ablation experiments to replace/strip the registry
        if tool_registry_override is not None:
            self._tool_registry = tool_registry_override
            self._tool_functions = list(tool_registry_override.values())

    # Read-only tools available for random selection
    _READ_TOOLS = ("get_link_stats", "get_network_load")

    def _run_agent(self, event_description: str) -> AgentDecision:
        record = AgentDecision(
            sim_time=self.ctx.simulator.now,
            trigger=event_description,
            messages=[],
        )

        t0 = wallclock.perf_counter()

        # Simulate LLM latency
        sleep_s = self._mock_rng.uniform(0.01, 0.05)
        wallclock.sleep(sleep_s)

        # Call 1-3 read tools
        n_read = int(self._mock_rng.integers(1, 4))
        available_reads = [t for t in self._READ_TOOLS
                           if t in self._tool_registry]
        for _ in range(n_read):
            if not available_reads:
                break
            tool_name = available_reads[
                int(self._mock_rng.integers(0, len(available_reads)))
            ]
            try:
                self._tool_registry[tool_name]()
            except Exception:
                pass
            record.tool_calls_count += 1
            self.total_tool_calls += 1

        # Optionally call set_tx_params (80 % chance, if available)
        if "set_tx_params" in self._tool_registry and self._mock_rng.random() < 0.8:
            record.tool_calls_count += 1
            self.total_tool_calls += 1

            use_invalid = self._mock_rng.random() < 0.20   # 20 % invalid

            if use_invalid:
                # Inject an invalid argument
                invalid_choice = int(self._mock_rng.integers(0, 3))
                if invalid_choice == 0:
                    result = self._tool_registry["set_tx_params"](
                        device_ids=[0],
                        sf=6,  # invalid sf < 7
                        tx_power_dbm=14,
                        channel=868.1,
                        link="terrestrial",
                    )
                elif invalid_choice == 1:
                    result = self._tool_registry["set_tx_params"](
                        device_ids=[0],
                        sf=7,
                        tx_power_dbm=20,   # > 14 dBm
                        channel=868.1,
                        link="terrestrial",
                    )
                else:
                    result = self._tool_registry["set_tx_params"](
                        device_ids=[0],
                        sf=7,
                        tx_power_dbm=14,
                        channel=915.0,     # invalid channel (US915 in EU868)
                        link="terrestrial",
                    )

                if result.get("status") == "rejected":
                    record.invalid_calls_count += 1
                    self.total_invalid_calls += 1
                record.messages.append({"role": "tool", "content": str(result)})
            else:
                # Valid call
                sf = int(self._mock_rng.choice(list(VALID_SFS)))
                ch = float(self._mock_rng.choice(list(VALID_CHANNELS)))
                pwr = int(self._mock_rng.integers(2, 15))
                result = self._tool_registry["set_tx_params"](
                    device_ids=list(range(min(5, N_DEVICES))),
                    sf=sf,
                    tx_power_dbm=pwr,
                    channel=ch,
                    link="terrestrial",
                )
                record.messages.append({"role": "tool", "content": str(result)})

        record.wall_clock_latency_s = wallclock.perf_counter() - t0
        self.decisions.append(record)
        return record


# ═══════════════════════════════════════════════════════════════════════════
# Simulation setup helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_sim_and_ctx(seed: int = SEED) -> tuple[Simulator, SimulatorContext]:
    """Create a simulator with ~20 ADR packets pre-loaded and return ctx."""
    sim = Simulator(
        duration_s=DURATION_S,
        seed=seed,
        payload_bytes=20,
    )

    detector = RollingZScoreDetector(threshold=3.0)
    anomaly_schedule = AnomalySchedule()

    ctx = SimulatorContext(
        simulator=sim,
        detector=detector,
        anomaly_schedule=anomaly_schedule,
        satellite_pass=None,
    )

    # Pre-load ~20 ADR packets at realistic SNR
    sim.register_hook(EventType.SCHEDULE_TX, _default_schedule_hook)
    generate_poisson_traffic(
        sim,
        n_devices=N_DEVICES,
        mean_interval_s=300.0,
        sf=9,
        tx_power_dbm=14,
        link="terrestrial",
        snr_db=5.0,
        rx_power_dbm=-100.0,
    )

    return sim, ctx


def inject_schedule_tx_triggers(sim: Simulator, n: int = N_DECISIONS) -> None:
    """Inject N SCHEDULE_TX events spread across the simulation duration."""
    rng = np.random.default_rng(SEED + 1)
    times = np.sort(rng.uniform(100.0, DURATION_S - 100.0, size=n))
    for t in times:
        sim.schedule_event(float(t), EventType.SCHEDULE_TX, {
            "device_id": 0,
            "sf": 9,
            "tx_power_dbm": 14,
            "channel": 868.1,
            "link": "terrestrial",
            "rx_power_dbm": -100.0,
            "snr_db": 5.0,
            "doppler_rate": 0.0,
        })


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_e5() -> None:
    print("=" * 65)
    print("  E5 — Agent Cost Measurement (mock mode)")
    print("=" * 65)
    print(f"  N_devices={N_DEVICES}, duration={DURATION_S}s, seed={SEED}")
    print(f"  NOTE: Real latency requires a live ollama server.")
    print(f"        Mock latency reported here is a baseline floor.")
    print()

    rng = np.random.default_rng(SEED)
    sim, ctx = build_sim_and_ctx(seed=SEED)
    inject_schedule_tx_triggers(sim, n=N_DECISIONS)

    adr = ADRPolicy()
    trigger_checker = TriggerChecker(
        per_threshold=0.10,
        review_interval_s=1800.0,
    )

    agent = MockLLMAgentPolicy(
        ctx=ctx,
        rng=rng,
        adr=adr,
        trigger_checker=trigger_checker,
    )

    # ── drive the agent directly (bypass the simulator loop) ──────────
    # We want exactly N_DECISIONS measured decisions.
    # Advance simulated time manually to simulate a 2-hour window.
    decision_times = np.sort(
        np.random.default_rng(SEED + 2).uniform(10.0, DURATION_S - 10.0, N_DECISIONS)
    )

    rows = []
    print(f"  Running {N_DECISIONS} mock agent decisions …")
    for i, t in enumerate(decision_times):
        sim.now = float(t)
        trigger = f"SCHEDULE_TX event at t={t:.1f}s"
        record = agent._run_agent(trigger)

        rows.append({
            "decision_id": i,
            "latency_s": record.wall_clock_latency_s,
            "tool_calls": record.tool_calls_count,
            "invalid_calls": record.invalid_calls_count,
            "triggered_by": trigger,
        })

        if (i + 1) % 20 == 0:
            print(f"    … {i+1}/{N_DECISIONS} decisions done")

    print()

    # ── metrics ───────────────────────────────────────────────────────
    latencies = [r["latency_s"] for r in rows]
    tool_calls_all = [r["tool_calls"] for r in rows]
    invalid_all = [r["invalid_calls"] for r in rows]

    lat_median = float(np.median(latencies))
    lat_p95 = float(np.percentile(latencies, 95))
    mean_tool_calls = float(np.mean(tool_calls_all))
    total_set_tx = sum(
        1 for r in rows
        if r["tool_calls"] > (1 if r["invalid_calls"] == 0 else 0)
    )
    total_invalid = sum(r["invalid_calls"] for r in rows)
    # Compute invalid-action rate from agent counters directly
    total_set_tx_calls = agent.total_tool_calls - sum(
        min(r["tool_calls"] - r["invalid_calls"],
            max(r["tool_calls"] - 1, 0)) for r in rows
    )
    invalid_action_rate = (
        agent.total_invalid_calls / agent.total_tool_calls
        if agent.total_tool_calls > 0 else 0.0
    )
    decisions_per_hour = N_DECISIONS / (DURATION_S / 3600.0)

    print("  +-----------------------------------------+----------------+")
    print("  | Metric                                  | Value          |")
    print("  +-----------------------------------------+----------------+")
    print(f"  | Decisions simulated                     | {N_DECISIONS:<14d} |")
    print(f"  | Simulated duration (h)                  | {DURATION_S/3600:.1f}{'':10s} |")
    print(f"  | LLM calls / simulated hour (mock)       | {decisions_per_hour:<14.1f} |")
    print(f"  | Latency median (s)  [mock]              | {lat_median:<14.4f} |")
    print(f"  | Latency P95 (s)     [mock]              | {lat_p95:<14.4f} |")
    print(f"  | Mean tool calls / decision              | {mean_tool_calls:<14.2f} |")
    print(f"  | Total tool calls                        | {agent.total_tool_calls:<14d} |")
    print(f"  | Total invalid calls                     | {agent.total_invalid_calls:<14d} |")
    print(f"  | Invalid-action rate                     | {invalid_action_rate:<14.3f} |")
    print("  +-----------------------------------------+----------------+")
    print()
    print("  [Real latency with ollama qwen2.5:3b ~ 0.5-3 s; 7B ~ 1-6 s]")
    print()

    # -- save CSV ------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "e5_agent_cost.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["decision_id", "latency_s", "tool_calls",
                           "invalid_calls", "triggered_by"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV  -> {csv_path}")

    # -- save PNG ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram of latencies
    ax = axes[0]
    ax.hist(latencies, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(lat_median, color="orange", lw=2, linestyle="--",
               label=f"Median {lat_median*1e3:.1f} ms")
    ax.axvline(lat_p95, color="crimson", lw=2, linestyle=":",
               label=f"P95 {lat_p95*1e3:.1f} ms")
    ax.set_xlabel("Decision latency (s)  [mock]")
    ax.set_ylabel("Count")
    ax.set_title("E5 - Agent Decision Latency Distribution")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Tool calls per decision
    ax2 = axes[1]
    tc_values = tool_calls_all
    ax2.hist(tc_values, bins=range(0, max(tc_values) + 2), color="teal",
             edgecolor="white", alpha=0.85, align="left")
    ax2.set_xlabel("Tool calls per decision")
    ax2.set_ylabel("Count")
    ax2.set_title("E5 - Tool Calls per Decision")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    png_path = RESULTS_DIR / "e5_agent_cost.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"  Saved PNG  -> {png_path}")
    print()
    print("  Done - E5 complete.")


if __name__ == "__main__":
    run_e5()
