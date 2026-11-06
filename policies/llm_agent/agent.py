"""Event-triggered LLM agent on top of ADR (section 3.5 phase 6).

* ADR runs by default (base parameters of every device).
* On a trigger (detector alert, loss > 10 %, satellite pass event, periodic
  review every 30 min; 10-min cooldown per trigger), the agent is called
  with a short description of the trigger.  It may call up to
  ``max_tool_calls`` tools; ``set_tx_params`` issues leased overrides.
* Guardrails: JSON-schema-like validation of every ``set_tx_params`` call
  (rejected calls are counted), "read before you write" enforced (a write
  before any read is rejected), cap on tool calls, wall-clock timeout,
  fallback to ADR (no change) on any backend error.
* Simulated time is frozen while the agent thinks; the wall-clock latency of
  every LLM call is recorded (assumption A6).
* Every decision is saved (trigger, full message trace, tool calls and
  results, latencies) for the experiment log.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from policies.adr import ADRController
from policies.llm_agent.backends import BackendError
from policies.llm_agent.prompt import build_system_prompt
from policies.llm_agent.tools import READ_TOOLS, ToolError, call_tool, tool_schemas
from policies.triggers import TriggerManager


@dataclass
class AgentDecision:
    sim_time: float
    reasons: list[dict[str, Any]]
    messages: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    llm_latencies_s: list[float] = field(default_factory=list)
    cached_calls: int = 0
    tool_calls: int = 0
    set_calls: int = 0
    set_rejected: int = 0
    unknown_or_malformed_calls: int = 0
    read_before_write_violations: int = 0
    devices_targeted: int = 0
    fallback: bool = False
    error: str | None = None
    final_text: str = ""

    @property
    def latency_s(self) -> float:
        return float(sum(self.llm_latencies_s))


def _fmt_time(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def describe_trigger(now: float, reasons: list[dict[str, Any]]) -> str:
    """User message: the trigger category only, details must be read with tools."""
    labels = {
        "anomaly_alert": "the anomaly detector raised an alert",
        "high_loss": "the inferred loss rate exceeded 10 %",
        "satellite_rise": "the satellite has just risen above the minimum elevation",
        "satellite_set": "the satellite has just set below the minimum elevation",
        "satellite_set_soon": "the satellite will set in less than 2 minutes",
        "periodic_review": "periodic review (every 30 minutes)",
    }
    items = sorted({labels.get(r["trigger"], r["trigger"]) for r in reasons})
    return (f"Simulated time {_fmt_time(now)}. You are called because: " + "; ".join(items) +
            ". Inspect the network with the tools, then decide whether to change anything.")


class LLMAgentController(ADRController):
    name = "LLM-Agent"

    def __init__(self, backend: Any, agent_cfg: dict[str, Any] | None = None,
                 trace_path: str | Path | None = None, label: str | None = None) -> None:
        super().__init__()
        self.backend = backend
        self.agent_cfg = agent_cfg
        self.trace_path = Path(trace_path) if trace_path else None
        self.decisions: list[AgentDecision] = []
        if label:
            self.name = label

    def attach(self, sim: Any) -> None:
        super().attach(sim)
        self.acfg = self.agent_cfg or sim.cfg["agent"]
        self.trig = TriggerManager(self.acfg)
        self.tools = tool_schemas(sim.channels, include_anomaly=self.acfg["enable_anomaly_tool"])
        self.tool_names = {t["function"]["name"] for t in self.tools}
        self.system_prompt = build_system_prompt(sim.cfg, self.acfg["include_lora_facts"])
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text("")

    # ── decision loop ─────────────────────────────────────────────────

    def decide(self, sim: Any, reasons: list[dict[str, Any]]) -> AgentDecision:
        rec = AgentDecision(sim_time=sim.now, reasons=reasons)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": describe_trigger(sim.now, reasons)}]
        max_calls = int(self.acfg["max_tool_calls"])
        timeout = float(self.acfg["timeout_s"])
        has_read = False
        try:
            while rec.tool_calls < max_calls:
                resp = self.backend.chat(messages, self.tools)
                rec.llm_calls += 1
                rec.llm_latencies_s.append(float(resp["latency_s"]))
                rec.cached_calls += bool(resp.get("cached"))
                calls = resp["tool_calls"]
                amsg: dict[str, Any] = {"role": "assistant", "content": resp["content"]}
                if calls:
                    amsg["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                          for c in calls]
                messages.append(amsg)
                if not calls:
                    rec.final_text = resp["content"]
                    break
                for c in calls[: max_calls - rec.tool_calls]:
                    rec.tool_calls += 1
                    name, args = c["name"], c["arguments"] if isinstance(c["arguments"], dict) else {}
                    if name == "set_tx_params":
                        rec.set_calls += 1
                    try:
                        if name not in self.tool_names:
                            rec.unknown_or_malformed_calls += 1
                            raise ToolError(f"unknown tool '{name}'")
                        if name == "set_tx_params" and not has_read:
                            rec.read_before_write_violations += 1
                            raise ToolError("read before you write: call a get_ tool first")
                        result = call_tool(sim, name, args)
                        if name in READ_TOOLS:
                            has_read = True
                        if name == "set_tx_params":
                            rec.devices_targeted += int(result.get("n_devices", 0))
                    except (ToolError, TypeError, ValueError) as exc:
                        if name == "set_tx_params":
                            rec.set_rejected += 1
                        result = {"status": "rejected", "reason": str(exc)}
                    messages.append({"role": "tool", "content": json.dumps(result, default=str),
                                     "tool_name": name})
                # timeout on the measured LLM time (also correct on cached replays)
                if rec.latency_s > timeout:
                    rec.error = "timeout"
                    break
        except BackendError as exc:
            rec.fallback = True
            rec.error = str(exc)[:300]
        rec.messages = messages
        return rec

    def on_tick(self, sim: Any, now: float, alerts: list) -> None:
        reasons = self.trig.check(sim, now, alerts)
        if not reasons:
            return
        rec = self.decide(sim, reasons)
        self.decisions.append(rec)
        if self.trace_path:
            with self.trace_path.open("a") as f:
                d = asdict(rec)
                d["latency_s"] = rec.latency_s
                f.write(json.dumps(d, default=str) + "\n")

    # ── metrics (E5) ──────────────────────────────────────────────────

    def extra_metrics(self) -> dict[str, Any]:
        lat = [d.latency_s for d in self.decisions if not d.fallback]
        per_call = [x for d in self.decisions for x in d.llm_latencies_s]
        set_calls = sum(d.set_calls for d in self.decisions)
        rejected = sum(d.set_rejected for d in self.decisions)
        return {
            "agent_decisions": len(self.decisions),
            "agent_llm_calls": sum(d.llm_calls for d in self.decisions),
            "agent_tool_calls": sum(d.tool_calls for d in self.decisions),
            "agent_set_calls": set_calls,
            "agent_set_rejected": rejected,
            "invalid_action_rate": rejected / set_calls if set_calls else float("nan"),
            "agent_unknown_or_malformed": sum(d.unknown_or_malformed_calls for d in self.decisions),
            "agent_rbw_violations": sum(d.read_before_write_violations for d in self.decisions),
            "agent_fallbacks": sum(d.fallback for d in self.decisions),
            "agent_timeouts": sum(d.error == "timeout" for d in self.decisions),
            "decision_latency_median_s": float(np.median(lat)) if lat else float("nan"),
            "decision_latency_p95_s": float(np.percentile(lat, 95)) if lat else float("nan"),
            "llm_call_latency_median_s": float(np.median(per_call)) if per_call else float("nan"),
            "agent_cached_calls": sum(d.cached_calls for d in self.decisions),
            "adr_changes": self.changes,
        }
