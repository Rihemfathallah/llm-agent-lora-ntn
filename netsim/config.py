"""Load YAML configurations: base.yaml deep-merged with scenario overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: *override* merged into *base* (lists are replaced)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*names_or_paths: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load ``configs/base.yaml`` then merge each scenario file, then *overrides*.

    >>> cfg = load_config("s1_jamming", overrides={"network": {"n_devices": 100}})
    """
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    for item in names_or_paths:
        p = Path(item)
        if not p.suffix:
            p = CONFIG_DIR / f"{item}.yaml"
        elif not p.is_absolute() and not p.exists():
            p = CONFIG_DIR / p
        cfg = deep_merge(cfg, yaml.safe_load(p.read_text()) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


SCENARIOS = ["s1_jamming", "s2_deep_fade", "s3_pass_end", "s4_gateway_outage", "s5_traffic_surge"]
