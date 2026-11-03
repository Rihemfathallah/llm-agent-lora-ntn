"""LLM agent for adaptive modulation — ollama tool-calling loop.

Section 3.5 Phase 6 — architecture:
  - ADR runs by default; the agent is called ONLY on triggers.
  - Triggers: anomaly alert, PER > 10 %, satellite elevation crossing,
    periodic review every 30 simulated minutes.
  - Max 6 tool calls per decision; fallback to ADR on timeout or error.
  - Temperature 0, fixed seed for reproducibility.
  - Wall-clock latency is recorded but does NOT advance simulated time.
"""

from __future__ import annotations

import json
import logging
import time as wallclock
from dataclasses import dataclass, field
from typing import Any

from policies.fixed import BasePolicy, TxParams
from policies.adr import ADRPolicy
from policies.llm_agent.prompt import SYSTEM_PROMPT
from policies.llm_agent.tools import (
    SimulatorContext,
    make_tool_registry,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

MAX_TOOL_CALLS = 6              # guardrail: cap per decision
DEFAULT_MODEL = "qwen2.5:3b"   # also try llama3.2:3b or qwen2.5:7b
OLLAMA_OPTIONS = {"temperature": 0, "seed": 0}

# Trigger thresholds
PER_TRIGGER = 0.10              # PER over last window
PERIODIC_REVIEW_S = 1800.0      # 30 simulated minutes


# ═══════════════════════════════════════════════════════════════════════════
# Decision record
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentDecision:
    """Full record of one agent invocation (for experiment logging)."""
    sim_time: float
    trigger: str                          # why the agent was called
    messages: list[dict[str, Any]]        # full conversation trace
    tool_calls_count: int = 0
    invalid_calls_count: int = 0
    wall_clock_latency_s: float = 0.0
    fallback_to_adr: bool = False
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Trigger checker
# ═══════════════════════════════════════════════════════════════════════════

class TriggerChecker:
    """Evaluates whether the LLM agent should be invoked.

    Triggers (Section 3.5 Phase 6):
      1. The anomaly detector raises an alert.
      2. PER over the last packets exceeds 10 %.
      3. A satellite elevation threshold is crossed.
      4. A periodic review every 30 simulated minutes.
    """

    def __init__(
        self,
        per_threshold: float = PER_TRIGGER,
        review_interval_s: float = PERIODIC_REVIEW_S,
        elevation_threshold_deg: float = 15.0,
    ) -> None:
        self.per_threshold = per_threshold
        self.review_interval_s = review_interval_s
        self.elevation_threshold_deg = elevation_threshold_deg
        self._last_review_time: float = -review_interval_s
        self._last_elev_above: bool | None = None

    def check(self, ctx: SimulatorContext) -> str | None:
        """Return a trigger description string, or None if no trigger.

        Only the **first** matching trigger fires (priority order).
        """
        now = ctx.simulator.now

        # 1. Anomaly detector alert
        if ctx.detector and ctx.detector.alerts:
            recent = [a for a in ctx.detector.alerts
                      if a.time >= now - 60.0]
            if recent:
                a = recent[-1]
                return (f"Anomaly alert: {a.suspected_type} detected on "
                        f"channel {a.channel} (z={a.z_score:.1f}, "
                        f"confidence={a.confidence:.1f})")

        # 2. PER exceeds threshold
        recent_pkts = [p for p in ctx.simulator.packets
                       if p.end_time >= now - 300.0]
        if len(recent_pkts) >= 5:
            n_failed = sum(1 for p in recent_pkts if not p.delivered)
            per = n_failed / len(recent_pkts)
            if per > self.per_threshold:
                return (f"High PER: {per:.1%} over last "
                        f"{len(recent_pkts)} packets (threshold "
                        f"{self.per_threshold:.0%})")

        # 3. Satellite elevation crossing
        if ctx.satellite_pass and len(ctx.satellite_pass["t_s"]) > 0:
            import numpy as np
            elev = float(np.interp(now,
                                   ctx.satellite_pass["t_s"],
                                   ctx.satellite_pass["elevation_deg"],
                                   left=0.0, right=0.0))
            above = elev >= self.elevation_threshold_deg
            if self._last_elev_above is not None and above != self._last_elev_above:
                self._last_elev_above = above
                direction = "risen above" if above else "fallen below"
                return (f"Satellite elevation has {direction} "
                        f"{self.elevation_threshold_deg} deg "
                        f"(now {elev:.1f} deg)")
            self._last_elev_above = above

        # 4. Periodic review
        if now - self._last_review_time >= self.review_interval_s:
            self._last_review_time = now
            return f"Periodic review (every {self.review_interval_s/60:.0f} min)"

        return None


# ═══════════════════════════════════════════════════════════════════════════
# LLM Agent Policy
# ═══════════════════════════════════════════════════════════════════════════

class LLMAgentPolicy(BasePolicy):
    """Event-triggered LLM agent with ADR fallback.

    Architecture:
      - ADR runs continuously as the default policy.
      - On each simulation step the ``TriggerChecker`` is evaluated.
      - If a trigger fires, the LLM is called via ollama with the 5
        registered tools, capped at ``MAX_TOOL_CALLS`` rounds.
      - If ollama is unavailable or errors out, ADR handles the decision.
    """

    name = "LLM-Agent"

    def __init__(
        self,
        ctx: SimulatorContext,
        model: str = DEFAULT_MODEL,
        adr: ADRPolicy | None = None,
        trigger_checker: TriggerChecker | None = None,
    ) -> None:
        self.ctx = ctx
        self.model = model
        self.adr = adr or ADRPolicy()
        self.trigger_checker = trigger_checker or TriggerChecker()

        self._tool_registry = make_tool_registry(ctx)
        self._tool_functions = list(self._tool_registry.values())

        # Logging
        self.decisions: list[AgentDecision] = []
        self.total_tool_calls: int = 0
        self.total_invalid_calls: int = 0

    # ── core agent loop (ollama) ──────────────────────────────────────

    def _run_agent(self, event_description: str) -> AgentDecision:
        """Execute one LLM agent decision loop.

        Follows the loop sketch from Section 3.5 Phase 6:
          messages = [system, user]
          for _ in range(6):
              response = ollama.chat(...)
              if no tool_calls: break
              execute each tool, append results
          return messages
        """
        record = AgentDecision(
            sim_time=self.ctx.simulator.now,
            trigger=event_description,
            messages=[],
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": event_description},
        ]

        t0 = wallclock.perf_counter()

        try:
            import ollama as _ollama

            for _ in range(MAX_TOOL_CALLS):
                response = _ollama.chat(
                    model=self.model,
                    messages=messages,
                    tools=self._tool_functions,
                    options=OLLAMA_OPTIONS,
                )

                msg = response.message
                messages.append(_msg_to_dict(msg))

                if not msg.tool_calls:
                    break

                for call in msg.tool_calls:
                    fname = call.function.name
                    fargs = call.function.arguments

                    record.tool_calls_count += 1
                    self.total_tool_calls += 1

                    if fname not in self._tool_registry:
                        record.invalid_calls_count += 1
                        self.total_invalid_calls += 1
                        result = {"error": f"Unknown tool '{fname}'"}
                    else:
                        try:
                            result = self._tool_registry[fname](**fargs)
                        except (TypeError, ValueError) as exc:
                            record.invalid_calls_count += 1
                            self.total_invalid_calls += 1
                            result = {"error": str(exc)}

                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result, default=str),
                    })

        except ImportError:
            logger.warning("ollama library not installed — falling back to ADR")
            record.fallback_to_adr = True
            record.error = "ollama not installed"

        except Exception as exc:
            logger.warning("Agent error (%s) — falling back to ADR", exc)
            record.fallback_to_adr = True
            record.error = str(exc)

        record.wall_clock_latency_s = wallclock.perf_counter() - t0
        record.messages = messages
        self.decisions.append(record)
        return record

    # ── policy interface ──────────────────────────────────────────────

    def decide(self, device_id: int, context: dict | None = None) -> TxParams:
        """Return TX params, calling the LLM agent if a trigger fires.

        If no trigger fires or the agent errors out, the underlying ADR
        policy decides.
        """
        # Check if a trigger fires
        trigger = self.trigger_checker.check(self.ctx)

        if trigger is not None:
            record = self._run_agent(trigger)

            if not record.fallback_to_adr:
                # Check if the agent set params for this device
                if device_id in self.ctx.device_params:
                    p = self.ctx.device_params[device_id]
                    return TxParams(
                        sf=p["sf"],
                        tx_power_dbm=p["tx_power_dbm"],
                        channel=p["channel"],
                        link=p["link"],
                    )

        # Default: ADR decides
        return self.adr.decide(device_id, context)

    def update(self, device_id: int, snr_db: float, delivered: bool) -> None:
        """Forward feedback to the underlying ADR policy."""
        self.adr.update(device_id, snr_db, delivered)

    def reset(self) -> None:
        """Clear agent and ADR state."""
        self.adr.reset()
        self.decisions.clear()
        self.ctx.device_params.clear()
        self.total_tool_calls = 0
        self.total_invalid_calls = 0

    # ── metrics ───────────────────────────────────────────────────────

    def agent_metrics(self) -> dict[str, Any]:
        """Return agent-specific metrics (for experiment E5)."""
        latencies = [d.wall_clock_latency_s for d in self.decisions
                     if not d.fallback_to_adr]

        import numpy as np

        return {
            "total_decisions": len(self.decisions),
            "llm_decisions": sum(1 for d in self.decisions
                                 if not d.fallback_to_adr),
            "adr_fallbacks": sum(1 for d in self.decisions
                                 if d.fallback_to_adr),
            "total_tool_calls": self.total_tool_calls,
            "total_invalid_calls": self.total_invalid_calls,
            "invalid_action_rate": (
                self.total_invalid_calls / self.total_tool_calls
                if self.total_tool_calls > 0 else 0.0
            ),
            "latency_median_s": (
                float(np.median(latencies)) if latencies else None
            ),
            "latency_p95_s": (
                float(np.percentile(latencies, 95)) if latencies else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _msg_to_dict(msg: Any) -> dict[str, Any]:
    """Convert an ollama Message object to a plain dict for logging."""
    d: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in msg.tool_calls
        ]
    return d
