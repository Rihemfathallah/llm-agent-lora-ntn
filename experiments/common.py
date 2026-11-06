"""Shared helpers for experiments E3–E6: policy factory, runner, CSV I/O, stats."""

from __future__ import annotations

import os

# One OpenMP thread: single-row HistGradientBoosting predictions are ~30x
# faster without thread spin-up, and results do not depend on it.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netsim.config import load_config                       # noqa: E402
from netsim.simulator import NetworkSimulator, recovery_time  # noqa: E402

RESULTS = ROOT / "results"
TRACES = RESULTS / "traces"
ML_DIR = RESULTS / "ml"

BASELINES = ["Fixed-SF7", "Fixed-SF12", "ADR", "ML-HGB", "Oracle", "Triggered-Rules"]
AGENTS = ["LLM-Agent", "LLM-Agent-noAnomalyTool", "LLM-Agent-noFacts"]
COLORS = {"Fixed-SF7": "#8c8c8c", "Fixed-SF12": "#4d4d4d", "ADR": "#1f77b4",
          "ML-HGB": "#2ca02c", "Oracle": "#000000", "Triggered-Rules": "#ff7f0e",
          "LLM-Agent": "#d62728", "LLM-Agent-noAnomalyTool": "#e377c2",
          "LLM-Agent-noFacts": "#9467bd"}
# line styles so that curves that coincide (e.g. ADR and Triggered-Rules
# without any event) stay visible
STYLES = {"Triggered-Rules": "--", "LLM-Agent": ":", "LLM-Agent-noAnomalyTool": ":",
          "LLM-Agent-noFacts": ":"}


# ── LLM settings ─────────────────────────────────────────────────────────

def add_llm_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--backend", default=None, help="ollama | llamacpp (default: config)")
    ap.add_argument("--model", default=None, help="e.g. qwen2.5:3b (ollama) or a label (llamacpp)")
    ap.add_argument("--model-path", default=None, help="GGUF file for the llamacpp backend")
    ap.add_argument("--replay-only", action="store_true",
                    help="use only cached LLM answers (no model needed); fails on a cache miss")


def llm_overrides(args: argparse.Namespace) -> dict[str, Any]:
    o: dict[str, Any] = {}
    for k_arg, k_cfg in (("backend", "backend"), ("model", "model"), ("model_path", "model_path")):
        v = getattr(args, k_arg, None)
        if v is not None:
            o[k_cfg] = v
    return o


def model_slug(cfg: dict[str, Any]) -> str:
    return str(cfg["agent"]["model"]).replace(":", "-").replace("/", "-")


# ── ML model ─────────────────────────────────────────────────────────────

def get_ml_model(verbose: bool = True):
    """Train once on oracle labels from the training seeds; cache to disk."""
    from policies.ml_classifier import collect_training_data, load_model, save_model, train_model
    path = ML_DIR / "hgb_oracle_labels.joblib"
    if path.exists():
        return load_model(path)
    base = load_config()
    cfgs = [load_config("e3_scalability")] + [load_config(s) for s in
                                              ["s1_jamming", "s2_deep_fade", "s3_pass_end",
                                               "s4_gateway_outage", "s5_traffic_surge"]]
    t0 = time.time()
    X, y = collect_training_data(cfgs, base["train_seeds"])
    model = train_model(X, y, seed=0)
    save_model(model, path)
    meta = {"n_samples": int(len(y)), "n_classes": int(len(set(y.tolist()))),
            "train_seeds": base["train_seeds"], "train_accuracy": float(model.score(X, y)),
            "scenarios": ["e3_scalability", "s1", "s2", "s3", "s4", "s5"],
            "seconds": round(time.time() - t0, 1)}
    (ML_DIR / "hgb_training.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print("  ML model trained:", meta)
    return model


# ── policy factory ───────────────────────────────────────────────────────

def make_controller(name: str, cfg: dict[str, Any], trace_path: Path | None = None,
                    replay_only: bool = False):
    from policies.adr import ADRController
    from policies.fixed import FixedSF7, FixedSF12MaxPower
    from policies.oracle import OracleController
    from policies.rules import RulesController
    if name == "Fixed-SF7":
        return FixedSF7()
    if name == "Fixed-SF12":
        return FixedSF12MaxPower()
    if name == "ADR":
        return ADRController()
    if name == "Oracle":
        return OracleController()
    if name == "Triggered-Rules":
        return RulesController()
    if name == "ML-HGB":
        from policies.ml_classifier import MLController
        return MLController(get_ml_model())
    if name.startswith("LLM-Agent"):
        from policies.llm_agent.agent import LLMAgentController
        from policies.llm_agent.backends import make_backend
        acfg = dict(cfg["agent"])
        if name == "LLM-Agent-noAnomalyTool":
            acfg["enable_anomaly_tool"] = False
        elif name == "LLM-Agent-noFacts":
            acfg["include_lora_facts"] = False
        backend = make_backend(acfg, replay_only=replay_only)
        return LLMAgentController(backend, acfg, trace_path=trace_path, label=name)
    raise ValueError(f"unknown policy {name}")


# ── one run ──────────────────────────────────────────────────────────────

def run_one(cfg: dict[str, Any], seed: int, policy: str, exp: str, scenario: str,
            replay_only: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    trace = None
    if policy.startswith("LLM-Agent"):
        trace = TRACES / exp / f"{scenario}__{policy}__{model_slug(cfg)}__seed{seed}.jsonl"
    ctrl = make_controller(policy, cfg, trace, replay_only)
    t0 = time.time()
    sim = NetworkSimulator(cfg, seed, ctrl)
    summ = sim.run()
    wall = time.time() - t0
    ts = sim.time_series()
    row: dict[str, Any] = {"experiment": exp, "scenario": scenario, "policy": policy, "seed": seed,
                           "n_devices": cfg["network"]["n_devices"],
                           "receiver": cfg["phy"]["receiver"], "wall_s": round(wall, 2)}
    if policy.startswith("LLM-Agent"):
        row["model"] = cfg["agent"]["model"]
        row["backend"] = cfg["agent"]["backend"]
    row.update(summ)
    row.update(ctrl.extra_metrics())
    if cfg.get("anomalies"):
        a = cfg["anomalies"][0]
        t_ev = float(a["start_s"])
        t_end = t_ev + float(a.get("duration_s", 0.0))
        row["recovery_time_s"] = recovery_time(ts, t_ev, cfg, t_end)
        pre = sim.window_metrics(t_ev - cfg["metrics"]["pre_event_window_s"], t_ev)
        ev = sim.window_metrics(t_ev, t_end + 600.0 if t_end > t_ev else t_ev + 1800.0)
        post = sim.window_metrics(t_ev, sim.duration)
        row.update({"pdr_pre": pre["pdr"], "pdr_event": ev["pdr"], "sent_event": ev["sent"],
                    "delivered_event": ev["delivered"], "energy_event_j": ev["energy_j"],
                    "energy_per_delivered_event_mj": ev["energy_per_delivered_mj"],
                    "pdr_post": post["pdr"]})
    return row, ts


# ── CSV helpers ──────────────────────────────────────────────────────────

def append_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open() as f:
            existing = list(csv.DictReader(f))
    keys: list[str] = []
    for r in existing + rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in existing + rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in keys})


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.6g}"
    return v


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", None) else float("nan")
                if k in ("seed", "n_devices") and not math.isnan(r[k]):
                    r[k] = int(r[k])
            except ValueError:
                pass
    return rows


def done_keys(path: Path, fields: tuple[str, ...]) -> set[tuple]:
    if not path.exists():
        return set()
    out = set()
    for r in read_rows(path):
        out.add(tuple(str(int(r[f])) if f == "seed" or f == "n_devices" else str(r[f]) for f in fields))
    return out


def mean_std(vals: Iterable[float]) -> tuple[float, float, int]:
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def fmt_ms(vals: Iterable[float], scale: float = 1.0, digits: int = 1) -> str:
    m, s, n = mean_std(vals)
    if n == 0:
        return "n/a"
    return f"{m*scale:.{digits}f} ± {s*scale:.{digits}f}"
