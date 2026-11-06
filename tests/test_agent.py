"""LLM agent plumbing, tested with a scripted backend (test double, no LLM).

These tests check parsing, validation, guardrails and caching.  They say
nothing about the quality of an LLM's decisions; that is measured in E4–E6.
"""
import json

import pytest

from netsim.config import load_config
from netsim.simulator import NetworkSimulator
from policies.llm_agent.agent import LLMAgentController, describe_trigger
from policies.llm_agent.backends import BackendError, CachedBackend, parse_tool_calls
from policies.llm_agent.prompt import build_system_prompt
from policies.llm_agent.tools import ToolError, tool_schemas, validate_set

THR = {"phy": {"receiver": "threshold"}, "network": {"n_devices": 60}}


class ScriptedBackend:
    """Test double: returns a fixed sequence of responses."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append(messages)
        if not self.script:
            return {"content": "done", "tool_calls": [], "latency_s": 0.01}
        return dict(self.script.pop(0), latency_s=0.01)


def _sim(ctrl, scen="s1_jamming"):
    return NetworkSimulator(load_config(scen, overrides=THR), 0, ctrl)


def test_parse_tool_calls_qwen_format():
    txt = ('<think>x</think>ok <tool_call>{"name": "get_network_load", "arguments": {}}</tool_call>'
           '<tool_call>{bad json}</tool_call>')
    content, calls = parse_tool_calls(txt)
    assert content == "ok"
    assert calls[0] == {"name": "get_network_load", "arguments": {}}
    assert calls[1]["name"] == "__malformed__"


def test_tool_schema_names_match_spec():
    names = [t["function"]["name"] for t in tool_schemas([868.1, 868.3, 868.5])]
    assert names == ["get_link_stats", "get_satellite_status", "get_anomaly_report",
                     "get_network_load", "set_tx_params"]
    assert "get_anomaly_report" not in [t["function"]["name"] for t in
                                        tool_schemas([868.1], include_anomaly=False)]


def test_validation_rejects_out_of_range():
    sim = _sim(LLMAgentController(ScriptedBackend([])))
    for bad in ({"target": "all", "sf": 13}, {"target": "all", "tx_power_dbm": 20},
                {"target": "all", "channels": [869.0]}, {"target": "all", "link": "moon"},
                {"target": "nobody", "sf": 9}, {"target": "all"},
                {"target": "all", "sf": 9, "duration_s": 10}, {"target": "all", "sf": 9.5},
                {"target": "all", "link": "satellite"}, {"target": "all", "sf": 9, "foo": 1}):
        with pytest.raises(ToolError):
            validate_set(sim, bad)
    ids, fields, lease = validate_set(sim, {"target": "device:1,2", "sf": "9", "channels": 868.3})
    assert ids == [1, 2] and fields == {"sf": 9, "channels": (868.3,)} and lease == 1800.0


def test_read_before_write_and_invalid_action_rate():
    write = {"content": "", "tool_calls": [{"name": "set_tx_params",
                                            "arguments": {"target": "all", "sf": 9}}]}
    read = {"content": "", "tool_calls": [{"name": "get_network_load", "arguments": {}}]}
    bad = {"content": "", "tool_calls": [{"name": "set_tx_params",
                                          "arguments": {"target": "all", "sf": 99}}]}
    ag = LLMAgentController(ScriptedBackend([write, read, bad, write]))
    sim = _sim(ag)
    d = ag.decide(sim, [{"trigger": "periodic_review"}])
    assert d.read_before_write_violations == 1
    assert d.set_calls == 3 and d.set_rejected == 2
    assert ag.extra_metrics() is not None
    assert len(sim.pending) == len(sim.devices)        # the last write was applied


def test_tool_cap_and_unknown_tool():
    spam = {"content": "", "tool_calls": [{"name": "hack_the_planet", "arguments": {}}] * 10}
    ag = LLMAgentController(ScriptedBackend([spam]))
    sim = _sim(ag)
    d = ag.decide(sim, [{"trigger": "periodic_review"}])
    assert d.tool_calls == 6 and d.unknown_or_malformed_calls == 6


def test_backend_error_falls_back_to_adr():
    class Broken:
        def chat(self, m, t):
            raise BackendError("down")
    ag = LLMAgentController(Broken())
    sim = _sim(ag)
    d = ag.decide(sim, [{"trigger": "periodic_review"}])
    assert d.fallback and not sim.pending


def test_cache_replays_identical_calls(tmp_path):
    inner = ScriptedBackend([{"content": "a", "tool_calls": []}])
    cb = CachedBackend(inner, tmp_path, "test", "m", {"temperature": 0})
    r1 = cb.chat([{"role": "user", "content": "x"}], [])
    cb2 = CachedBackend(None, tmp_path, "test", "m", {"temperature": 0})
    r2 = cb2.chat([{"role": "user", "content": "x"}], [])
    assert r1["content"] == r2["content"] and r2["cached"] and r2["latency_s"] == r1["latency_s"]
    with pytest.raises(BackendError):
        cb2.chat([{"role": "user", "content": "other"}], [])


def test_prompt_ablation_and_trigger_text():
    cfg = load_config()
    full, bare = build_system_prompt(cfg, True), build_system_prompt(cfg, False)
    assert "LoRa facts" in full and "LoRa facts" not in bare
    msg = describe_trigger(3660, [{"trigger": "anomaly_alert", "kind": "noise_rise"}])
    assert "01:01:00" in msg and "868" not in msg       # no detail leaks into the trigger
