"""Tools exposed to the LLM agent (section 3.5 phase 6).

| Tool                  | Returns / does                                          |
|-----------------------|---------------------------------------------------------|
| get_link_stats        | uplinks, inferred loss rate, mean/min SNR, SF / power / |
|                       | link mix, ToA, duty-cycle use, silent devices           |
| get_satellite_status  | elevation now and in 1/5/10 min, Doppler shift & rate,  |
|                       | time to pass end, next pass                             |
| get_anomaly_report    | detector output: type, start, channels, devices, conf.  |
| get_network_load      | uplinks per SF and per channel, collision rate          |
| set_tx_params         | validated; returns "applied" or "rejected" + reason     |

Everything returned is network-server-side information; nothing reads the
ground truth.  (Exception, stated: the collision rate in get_network_load is
the true one, standing in for what a gateway can estimate from overlapping
preambles.)

Design decisions that differ from the first specification (see
docs/ARCHITECTURE.md):

* ``set_tx_params`` takes a ``target`` group instead of a list of device
  ids: with 100–1000 devices a 3B model cannot enumerate ids.  Groups:
  ``all``, ``terrestrial``, ``satellite``, ``affected`` (devices the network
  server sees as degraded), or ``device:<id>[,<id>…]``.
* ``channels`` restricts the channels a device hops over (LoRaWAN channel
  mask) instead of fixing a single channel.
* Every change is an override with a lease (``duration_s``, 60–7200 s,
  default 1800 s): when it expires the device returns to ADR.  This removes
  the "overrides never expire" problem of the first version.
* Changes reach a device in the downlink after its next delivered uplink
  (class A) or by satellite broadcast during a pass.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from phy.css import SNR_THRESHOLDS_DB
from policies.triggers import affected_devices, ns_loss_rate

VALID_SFS = (7, 8, 9, 10, 11, 12)
POWER_RANGE = (2, 14)
VALID_LINKS = ("terrestrial", "satellite", "wait")
LEASE_RANGE = (60.0, 7200.0)


# ═══════════════════════════════════════════════════════════════════════════
# JSON schemas (what the model sees)
# ═══════════════════════════════════════════════════════════════════════════

def tool_schemas(channels: list[float], include_anomaly: bool = True) -> list[dict[str, Any]]:
    ch_txt = ", ".join(f"{c:.1f}" for c in channels)
    tools = [
        {"type": "function", "function": {
            "name": "get_link_stats",
            "description": "Uplink statistics seen by the network server over a recent window: "
                           "number of uplinks, loss rate inferred from frame-counter gaps, mean and "
                           "minimum SNR, mix of SF / power / link, mean time on air, silent devices.",
            "parameters": {"type": "object", "properties": {
                "scope": {"type": "string",
                          "description": "all | terrestrial | satellite | affected | device:<id>"},
                "window_s": {"type": "number", "description": "look-back window in seconds (default 600)"},
            }, "required": []}}},
        {"type": "function", "function": {
            "name": "get_satellite_status",
            "description": "LEO satellite gateway status from the ephemeris: visible or not, elevation "
                           "now and in 1, 5 and 10 minutes, Doppler shift and Doppler rate, time until "
                           "the pass ends, time until the next pass.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {
            "name": "get_anomaly_report",
            "description": "Output of the anomaly detector over the last 10 minutes: suspected type "
                           "(jamming, deep_fade, gateway_outage, traffic_surge), start time, affected "
                           "channels, affected devices, confidence, ongoing conditions.",
            "parameters": {"type": "object", "properties": {
                "scope": {"type": "string", "description": "all (default)"},
            }, "required": []}}},
        {"type": "function", "function": {
            "name": "get_network_load",
            "description": "Network load over the last 10 minutes: uplinks per spreading factor and per "
                           "channel, collision rate, duty-cycle drops.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {
            "name": "set_tx_params",
            "description": "Override the transmission parameters of a group of devices for a limited "
                           "time (then they return to ADR). Omitted fields keep their current value. "
                           "Takes effect at each device's next received uplink, or by satellite "
                           "broadcast during a pass. Returns applied or rejected with a reason.",
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string",
                           "description": "all | terrestrial | satellite | affected | device:<id>[,<id>...]"},
                "sf": {"type": "integer", "description": "spreading factor 7-12"},
                "tx_power_dbm": {"type": "integer", "description": "transmit power 2-14 dBm"},
                "channels": {"type": "array", "items": {"type": "number"},
                             "description": f"allowed channels in MHz, subset of [{ch_txt}]"},
                "link": {"type": "string", "description": "terrestrial | satellite | wait"},
                "duration_s": {"type": "number", "description": "lease 60-7200 s (default 1800)"},
            }, "required": ["target"]}}},
    ]
    if not include_anomaly:
        tools = [t for t in tools if t["function"]["name"] != "get_anomaly_report"]
    return tools


# ═══════════════════════════════════════════════════════════════════════════
# Implementations
# ═══════════════════════════════════════════════════════════════════════════

class ToolError(ValueError):
    pass


def resolve_target(sim: Any, target: Any) -> list[int]:
    if not isinstance(target, str) or not target.strip():
        raise ToolError("target must be a string: all | terrestrial | satellite | affected | device:<id>")
    t = target.strip().lower()
    if t == "all":
        return [d.device_id for d in sim.devices]
    if t in ("terrestrial", "satellite"):
        return [d.device_id for d in sim.devices if d.home_link == t]
    if t == "affected":
        return affected_devices(sim, sim.now)
    if t.startswith("device:"):
        try:
            ids = [int(x) for x in t[7:].split(",") if x.strip()]
        except ValueError as exc:
            raise ToolError(f"bad device list in target '{target}'") from exc
        bad = [i for i in ids if not 0 <= i < len(sim.devices)]
        if bad or not ids:
            raise ToolError(f"unknown device ids {bad or ids}")
        return ids
    raise ToolError(f"unknown target '{target}'")


def get_link_stats(sim: Any, scope: str = "all", window_s: float = 600.0) -> dict[str, Any]:
    window_s = float(min(max(float(window_s or 600.0), 60.0), 7200.0))
    ids = set(resolve_target(sim, scope or "all"))
    recs = [r for r in sim.ns.window(sim.now, window_s) if r.device_id in ids]
    lost = sum(r.lost_before for r in recs)
    snrs = [r.snr_db for r in recs]
    eff = [sim.devices[i].effective_params(sim.now) for i in ids]
    silent = [i for i in ids if sim.now > 2.5 * sim.interval
              and sim.now - sim.ns.last_seen.get(i, 0.0) > 2.5 * sim.interval]
    mean_toa = float(np.mean([sim.toa[p.sf] for p in eff])) if eff else None
    out = {
        "scope": scope, "window_s": window_s, "n_devices": len(ids),
        "uplinks_received": len(recs), "uplinks_lost_inferred": lost,
        "loss_rate": round(lost / (lost + len(recs)), 3) if recs or lost else None,
        "mean_snr_db": round(float(np.mean(snrs)), 1) if snrs else None,
        "min_snr_db": round(float(np.min(snrs)), 1) if snrs else None,
        "sf_mix": dict(sorted(Counter(p.sf for p in eff).items())),
        "power_mix_dbm": dict(sorted(Counter(p.tx_power_dbm for p in eff).items())),
        "link_mix": dict(Counter(p.link for p in eff)),
        "mean_time_on_air_s": round(mean_toa, 3) if mean_toa else None,
        "duty_cycle_off_time_per_packet_s": round(mean_toa * sim.duty_off, 1) if mean_toa else None,
        "silent_devices": len(silent),
        "active_overrides": sum(1 for i in ids if sim.devices[i].override is not None
                                and sim.now < sim.devices[i].override_until),
        "pending_overrides": sum(1 for i in ids if i in sim.pending),
    }
    if scope == "all" or scope is None:
        out["network_loss_rate_10min"] = round(ns_loss_rate(sim, sim.now)[0], 3)
    return out


def get_satellite_status(sim: Any) -> dict[str, Any]:
    if not sim.sat_enabled or not sim.passes:
        return {"satellite_configured": False}
    st = sim.satellite_state(sim.now)

    def elev(dt: float) -> float | None:
        s = sim.satellite_state(sim.now + dt)
        return round(s["elevation_deg"], 1) if s["visible"] else None

    out = {"satellite_configured": True, "visible": st["visible"],
           "min_elevation_deg": sim.min_elev,
           "elevation_now_deg": elev(0), "elevation_in_1min_deg": elev(60),
           "elevation_in_5min_deg": elev(300), "elevation_in_10min_deg": elev(600)}
    if st["visible"]:
        out.update({"doppler_shift_hz": round(st["doppler_hz"]),
                    "doppler_rate_hz_s": round(st["doppler_rate_hz_s"], 1),
                    "time_to_pass_end_s": round(st["time_to_set_s"])})
    nxt = [p for p in sim.passes if p.t_rise > sim.now]
    if nxt:
        out["next_pass_in_s"] = round(nxt[0].t_rise - sim.now)
        out["next_pass_max_elevation_deg"] = nxt[0].max_elevation
        out["next_pass_duration_s"] = round(2 * nxt[0].half_duration)
    else:
        out["next_pass_in_s"] = None
    return out


def get_anomaly_report(sim: Any, scope: str = "all") -> dict[str, Any]:
    rep = sim.detector.report(sim.now, 600.0)
    rep = dict(rep)
    rep["affected_device_ids"] = rep["affected_device_ids"][:20]
    return rep


def get_network_load(sim: Any) -> dict[str, Any]:
    pk = [p for p in sim.packets if p.end >= sim.now - 600.0]
    recs = sim.ns.window(sim.now, 600.0)
    return {
        "window_s": 600,
        "uplinks_received_per_sf": dict(sorted(Counter(r.sf for r in recs).items())),
        "uplinks_received_per_channel_mhz": dict(sorted(Counter(r.channel for r in recs).items())),
        "collision_rate": round(sum(p.collided for p in pk) / len(pk), 3) if pk else 0.0,
        "readings_dropped_by_duty_cycle": int(sum(d.stats["dropped_duty"] for d in sim.devices)),
    }


def _as_int(name: str, v: Any) -> int:
    if isinstance(v, bool):
        raise ToolError(f"{name} must be an integer")
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{name} must be an integer, got {v!r}") from exc
    if not math.isfinite(f) or f != int(f):
        raise ToolError(f"{name} must be an integer, got {v!r}")
    return int(f)


def validate_set(sim: Any, args: dict[str, Any]) -> tuple[list[int], dict[str, Any], float]:
    """JSON-schema-like validation of set_tx_params. Raises ToolError."""
    allowed = {"target", "sf", "tx_power_dbm", "channels", "link", "duration_s"}
    extra = set(args) - allowed
    if extra:
        raise ToolError(f"unknown argument(s) {sorted(extra)}")
    ids = resolve_target(sim, args.get("target"))
    if not ids:
        raise ToolError("target group is empty")
    fields: dict[str, Any] = {}
    if args.get("sf") is not None:
        sf = _as_int("sf", args["sf"])
        if sf not in VALID_SFS:
            raise ToolError(f"sf must be 7-12, got {sf}")
        fields["sf"] = sf
    if args.get("tx_power_dbm") is not None:
        p = _as_int("tx_power_dbm", args["tx_power_dbm"])
        if not POWER_RANGE[0] <= p <= POWER_RANGE[1]:
            raise ToolError(f"tx_power_dbm must be 2-14, got {p}")
        fields["tx_power_dbm"] = p
    if args.get("channels") is not None:
        ch = args["channels"]
        if isinstance(ch, (int, float, str)):
            ch = [ch]
        try:
            ch = [float(c) for c in ch]
        except (TypeError, ValueError) as exc:
            raise ToolError(f"channels must be a list of numbers, got {args['channels']!r}") from exc
        valid = [c for c in sim.channels]
        bad = [c for c in ch if not any(abs(c - v) < 1e-6 for v in valid)]
        if bad or not ch:
            raise ToolError(f"channels must be a non-empty subset of {valid}, got {args['channels']!r}")
        fields["channels"] = tuple(sorted(set(ch)))
    if args.get("link") is not None:
        link = str(args["link"]).strip().lower()
        if link not in VALID_LINKS:
            raise ToolError(f"link must be terrestrial | satellite | wait, got {args['link']!r}")
        if link == "satellite" and not (sim.sat_enabled and sim.passes):
            raise ToolError("no satellite gateway in this deployment")
        fields["link"] = link
    if not fields:
        raise ToolError("nothing to change: give at least one of sf, tx_power_dbm, channels, link")
    lease = args.get("duration_s")
    lease = float(sim.cfg["agent"]["default_lease_s"]) if lease is None else float(lease)
    if not LEASE_RANGE[0] <= lease <= LEASE_RANGE[1]:
        raise ToolError(f"duration_s must be 60-7200, got {lease}")
    return ids, fields, lease


def set_tx_params(sim: Any, **args: Any) -> dict[str, Any]:
    ids, fields, lease = validate_set(sim, args)
    sim.issue_override(ids, fields, lease)
    return {"status": "applied", "n_devices": len(ids), "fields": {k: (list(v) if isinstance(v, tuple) else v)
                                                                   for k, v in fields.items()},
            "lease_s": lease,
            "note": "delivered at each device's next received uplink or by satellite broadcast"}


READ_TOOLS = {"get_link_stats", "get_satellite_status", "get_anomaly_report", "get_network_load"}


def call_tool(sim: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "get_link_stats":
        return get_link_stats(sim, **{k: v for k, v in args.items() if k in ("scope", "window_s")})
    if name == "get_satellite_status":
        return get_satellite_status(sim)
    if name == "get_anomaly_report":
        return get_anomaly_report(sim)
    if name == "get_network_load":
        return get_network_load(sim)
    if name == "set_tx_params":
        return set_tx_params(sim, **args)
    raise ToolError(f"unknown tool '{name}'")


def lora_facts(cfg: dict[str, Any]) -> str:
    """Table of LoRa facts computed from the code (no hand-written numbers)."""
    from phy.css import fft_bin_width, time_on_air
    from netsim.energy import tx_energy_joules
    rows = ["| SF | required SNR (dB) | time on air 20 B (ms) | energy at 14 dBm (mJ) | FFT bin (Hz) |",
            "|---|---|---|---|---|"]
    for sf in VALID_SFS:
        toa = time_on_air(cfg["phy"]["payload_bytes"], sf)
        rows.append(f"| {sf} | {SNR_THRESHOLDS_DB[sf]:.1f} | {toa*1e3:.1f} | "
                    f"{tx_energy_joules(14, toa)*1e3:.1f} | {fft_bin_width(sf):.1f} |")
    return "\n".join(rows)
