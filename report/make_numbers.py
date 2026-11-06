#!/usr/bin/env python
"""Generate report/numbers.tex: every number quoted in the report, as a LaTeX
macro computed from results/*.csv.  The report never contains a typed number.

Run: python report/make_numbers.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from common import mean_std, read_rows  # noqa: E402

RES = ROOT / "results"
OUT = Path(__file__).with_name("numbers.tex")
macros: dict[str, str] = {}


def put(name: str, value) -> None:
    assert name.isalpha(), name
    macros[name] = str(value)


def rd(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{100 * x:.{d}f}"


def num(x: float, d: int = 1) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.{d}f}"


def main() -> None:
    # E1
    th = rd(RES / "e1a_thresholds.csv")
    gaps = [float(r["gap_per1pct_vs_threshold_db"]) for r in th]
    put("EoneGapMin", f"{min(gaps):.2f}")
    put("EoneGapMax", f"{max(gaps):.2f}")
    put("EoneGapSFtwelve", next(r["gap_per1pct_vs_threshold_db"] for r in th if r["sf"] == "12"))
    mc = rd(RES / "e1a_ser_vs_snr_awgn.csv")
    put("EoneMCTrials", mc[0]["n_trials"])
    put("EoneMCMaxDiff", f"{max(abs(float(r['ser_mc']) - float(r['ser_analytic'])) for r in mc):.3f}")
    cr = rd(RES / "e1b_critical_rate.csv")
    g = lambda rx, sf, k: next(r[k] for r in cr if r["receiver"] == rx and r["sf"] == str(sf))
    for rx, tag in (("preamble", "A"), ("tracking", "B")):
        for sf, w in ((9, "nine"), (10, "ten"), (11, "eleven"), (12, "twelve")):
            put(f"Crit{tag}SF{w}", g(rx, sf, "critical_rate_hz_s").replace(">", "$>$"))
            put(f"Ok{tag}SF{w}", f"{float(g(rx, sf, 'last_ok_rate_hz_s')):.0f}")
    put("TrackLimitSFtwelve", f"{float(g('tracking', 12, 'tracking_limit_theory_hz_s')):.0f}")
    put("DriftBinsSFtwelve", g("preamble", 12, "drift_bins_at_252hz_s_preamble_only"))
    put("PerPackets", g("tracking", 12, "n_packets_per_point"))

    # E2
    e2 = rd(RES / "e2_satellite_pass.csv")
    put("EtwoDuration", f"{(len(e2) - 1) / 60:.1f}")
    put("EtwoMaxDoppler", f"{max(abs(float(r['doppler_hz'])) for r in e2) / 1e3:.1f}")
    put("EtwoMaxRate", f"{max(abs(float(r['doppler_rate_hz_s'])) for r in e2):.0f}")
    put("EtwoSnrMax", f"{max(float(r['snr_los_db']) for r in e2):.1f}")
    put("EtwoSnrMin", f"{min(float(r['snr_los_db']) for r in e2):.1f}")
    put("EtwoDelayMin", f"{min(float(r['delay_ms']) for r in e2):.1f}")
    put("EtwoDelayMax", f"{max(float(r['delay_ms']) for r in e2):.1f}")
    alt = rd(RES / "e2b_altitude_sweep.csv")
    ok12 = sorted(int(r["altitude_km"]) for r in alt if r["receiver"] == "tracking" and r["sf"] == "12"
                  and r["feasible_at_zenith"] == "1")
    put("EtwoBSFtwelveZenithFrom", ok12[0] if ok12 else "n/a")
    sh = lambda rx, sf, h, k="feasible_share": float(next(r[k] for r in alt if r["receiver"] == rx and r["sf"] == str(sf) and r["altitude_km"] == str(h)))
    put("EtwoAshareSFtwelve", pct(sh("preamble", 12, 600), 0))
    put("EtwoAshareSFtwelveSnr", pct(sh("preamble", 12, 600, "feasible_share_snr_only"), 0))
    put("EtwoBshareSFtwelve", pct(sh("tracking", 12, 600), 0))
    put("EtwoRateFiveFifty", next(r["zenith_doppler_rate_hz_s"] for r in alt if r["altitude_km"] == "550"))

    # E3
    e3 = read_rows(RES / "e3_scalability.csv")
    al = read_rows(RES / "e3_aloha_check.csv")
    put("AlohaMaxErr", f"{max(abs(r['S_sim'] - r['S_theory']) for r in al):.4f}")
    put("AlohaMaxG", f"{max(r['G'] for r in al):.2f}")
    ns = sorted({r["n_devices"] for r in e3})
    for pol, tag in (("Fixed-SF7", "SFseven"), ("Fixed-SF12", "SFtwelve"), ("ADR", "ADR"),
                     ("ML-HGB", "ML"), ("Oracle", "Oracle"), ("Triggered-Rules", "Rules"), ("LLM-Agent", "Agent")):
        for n, nt in ((ns[0], "Lo"), (ns[-1], "Hi")):
            sel = [r for r in e3 if r["policy"] == pol and r["n_devices"] == n]
            if sel:
                m, s, _ = mean_std(r["pdr"] for r in sel)
                put(f"EthreePdr{tag}{nt}", pct(m))
                put(f"EthreePdrSd{tag}{nt}", pct(s))
                m, s, _ = mean_std(r["energy_per_delivered_mj"] for r in sel)
                put(f"EthreeE{tag}{nt}", num(m))
    put("EthreeNlo", ns[0])
    put("EthreeNhi", ns[-1])

    # E4
    e4 = read_rows(RES / "e4_summary.csv")
    scen = ["s1_jamming", "s2_deep_fade", "s3_pass_end", "s4_gateway_outage", "s5_traffic_surge"]
    stag = dict(zip(scen, ["One", "Two", "Three", "Four", "Five"]))
    ptag = {"Fixed-SF7": "SFseven", "Fixed-SF12": "SFtwelve", "ADR": "ADR", "ML-HGB": "ML",
            "Oracle": "Oracle", "Triggered-Rules": "Rules", "LLM-Agent": "Agent"}
    for s in scen:
        for p, t in ptag.items():
            sel = [r for r in e4 if r["scenario"] == s and r["policy"] == p]
            if not sel:
                continue
            m, sd, _ = mean_std(r["pdr_event"] for r in sel)
            put(f"S{stag[s]}Pdr{t}", pct(m))
            put(f"S{stag[s]}PdrSd{t}", pct(sd))
            m, _, _ = mean_std(r["energy_per_delivered_event_mj"] for r in sel)
            put(f"S{stag[s]}E{t}", num(m))
            m, _, _ = mean_std(r["reading_delivery_ratio"] for r in sel)
            put(f"S{stag[s]}Rdr{t}", pct(m))
            rec = [r["recovery_time_s"] for r in sel]
            ok = [x for x in rec if not math.isnan(x)]
            put(f"S{stag[s]}Rec{t}", f"{np.mean(ok) / 60:.0f}" if ok else "--")
            put(f"S{stag[s]}Cens{t}", str(len(rec) - len(ok)))
    # paired differences against ADR (event-window PDR, percentage points)
    for s in scen:
        for p, t in ptag.items():
            if p == "ADR":
                continue
            a = {r["seed"]: r["pdr_event"] for r in e4 if r["scenario"] == s and r["policy"] == p}
            b = {r["seed"]: r["pdr_event"] for r in e4 if r["scenario"] == s and r["policy"] == "ADR"}
            d = [a[k] - b[k] for k in a if k in b]
            if d:
                m, sd, n = mean_std(d)
                put(f"S{stag[s]}Diff{t}", f"{100 * m:+.1f}".replace("-", "$-$"))
                put(f"S{stag[s]}DiffSd{t}", f"{100 * sd:.1f}")
                put(f"S{stag[s]}Wins{t}", str(sum(1 for x in d if x > 1e-9)))
                put(f"S{stag[s]}Losses{t}", str(sum(1 for x in d if x < -1e-9)))
    det = [r for r in e4 if r["policy"] == "ADR" and r["scenario"] != "s3_pass_end"]
    put("DetPrecision", num(mean_std(r["det_precision"] for r in det)[0], 2))
    put("DetRecall", num(mean_std(r["det_recall"] for r in det)[0], 2))
    for s in scen:
        if s == "s3_pass_end":
            continue
        sel = [r for r in det if r["scenario"] == s]
        put(f"S{stag[s]}DetDelay", num(mean_std(r["det_detection_delay_s"] for r in sel)[0] / 60, 0))
        put(f"S{stag[s]}DetPrec", num(mean_std(r["det_precision"] for r in sel)[0], 2))

    # E5
    p5 = RES / "e5_agent_cost.csv"
    if p5.exists():
        e5 = rd(p5)
        main_row = next((r for r in e5 if r["policy"] == "LLM-Agent" and r["experiment"] == "e4"), None)
        if main_row:
            put("AgentModel", main_row["model"].replace("_", r"\_"))
            put("AgentDecisions", main_row["decisions"])
            put("AgentRuns", main_row["runs"])
            put("AgentCallsHour", main_row["llm_calls_per_sim_hour"])
            put("AgentDecHour", main_row["decisions_per_sim_hour"])
            put("AgentLatMed", main_row["decision_latency_median_s"])
            put("AgentLatPninetyfive", main_row["decision_latency_p95_s"])
            put("AgentCallLatMed", main_row["llm_call_latency_median_s"])
            put("AgentInvalid", pct(float(main_row["invalid_action_rate"]), 0) if main_row["invalid_action_rate"] != "nan" else "n/a")
            put("AgentSetCalls", main_row["set_calls"])
            put("AgentSetRejected", main_row["set_rejected"])
            put("AgentChanges", main_row["decisions_with_a_change"])
            put("AgentRbw", main_row["rbw_violations"])
            put("AgentToolsPerDec", main_row["tool_calls_per_decision"])
            put("AgentHost", main_row["host"])
    # agent actions per scenario (from the decision traces of E4)
    pdec = RES / "e5_decisions.csv"
    if pdec.exists():
        dec = [r for r in rd(pdec) if r["experiment"] == "e4" and r["policy"] == "LLM-Agent"]
        for s in scen:
            runs = {}
            for r in dec:
                if r["scenario"] == s:
                    runs.setdefault(r["seed"], 0)
                    runs[r["seed"]] += int(r["set_calls"]) - int(r["set_rejected"])
            put(f"S{stag[s]}AgentRunsActed", str(sum(1 for v in runs.values() if v > 0)))
        put("AgentAppliedTotal", str(sum(int(r["set_calls"]) - int(r["set_rejected"]) for r in dec)))
    # E6 ablations
    p6 = RES / "e6_summary.csv"
    if p6.exists():
        e6 = read_rows(p6)
        for p, t in (("LLM-Agent-noAnomalyTool", "NoTool"), ("LLM-Agent-noFacts", "NoFacts")):
            for s in scen:
                sel = [r for r in e6 if r["scenario"] == s and r["policy"] == p]
                if sel:
                    m, sd, _ = mean_std(r["pdr_event"] for r in sel)
                    put(f"S{stag[s]}Pdr{t}", pct(m))
                    put(f"S{stag[s]}PdrSd{t}", pct(sd))
            sets = sum(r.get("agent_set_calls", 0) for r in e6 if r["policy"] == p)
            rej = sum(r.get("agent_set_rejected", 0) for r in e6 if r["policy"] == p)
            put(f"Applied{t}", str(int(sets - rej)))
            put(f"Decisions{t}", str(int(sum(r.get("agent_decisions", 0) for r in e6 if r["policy"] == p))))
    # E3 with the agent: paired differences to ADR, worst seed
    e3a = [r for r in e3 if r["policy"] == "LLM-Agent"]
    if e3a:
        diffs = []
        for r in e3a:
            b = next(x for x in e3 if x["policy"] == "ADR" and x["n_devices"] == r["n_devices"] and x["seed"] == r["seed"])
            diffs.append((r["pdr"] - b["pdr"], r["n_devices"]))
        worst = min(diffs)
        put("EthreeAgentWorst", f"{100 * worst[0]:.1f}".replace("-", "$-$"))
        put("EthreeAgentWorstN", str(worst[1]))
        put("EthreeAgentLossRuns", str(sum(1 for d, _ in diffs if d < -1e-9)))
        put("EthreeAgentRuns", str(len(diffs)))
        sel = [r for r in e3a if r["n_devices"] <= 200]
        put("EthreeAgentEqualUpTo", "200" if all(abs(d) < 1e-9 for d, n in diffs if n <= 200) else "n/a")
    # every change the agent applied (E3 + E4 traces)
    applied, template = 0, 0
    for tr in sorted((RES / "traces").glob("e[34]/*__LLM-Agent__*.jsonl")):
        for line in tr.read_text().splitlines():
            d = json.loads(line)
            msgs = d["messages"]
            calls = [c["function"]["arguments"] for m in msgs if m.get("tool_calls")
                     for c in m["tool_calls"] if c["function"]["name"] == "set_tx_params"]
            results = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"
                       and m.get("tool_name") == "set_tx_params"]
            for a, res in zip(calls, results):
                if res.get("status") == "applied":
                    applied += 1
                    template += int(a.get("target") == "all" and int(a.get("sf", 0)) == 12)
    put("AgentAppliedAll", str(applied))
    put("AgentAppliedTemplate", str(template))
    lines = ["% generated by report/make_numbers.py from results/*.csv -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT.name} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
