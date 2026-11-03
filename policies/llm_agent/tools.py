"""Tools exposed to the LLM agent for network control.

Section 3.5 Phase 6 — five tools that the agent can call:

| Tool                  | Returns / does                                        |
|-----------------------|-------------------------------------------------------|
| get_link_stats        | Mean/min SNR, PER, current SF, power, link, ToA,      |
|                       | remaining duty-cycle budget                           |
| get_satellite_status  | Elevation now and in 1/5/10 min, Doppler, pass end    |
| get_anomaly_report    | Detector output: type, start, channels, confidence    |
| get_network_load      | Packets per SF and channel, collision rate             |
| set_tx_params         | Validated by the simulator; "applied" or "rejected"   |

Each tool is a plain Python function with a docstring (used by the
ollama library for automatic tool-schema generation).  The functions
receive a ``SimulatorContext`` that holds references to the live
simulator, anomaly detector, satellite pass data, and policy state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from phy.css import SNR_THRESHOLDS_DB, time_on_air
from netsim.simulator import (
    Simulator, Packet, EU868_CHANNELS,
    DUTY_CYCLE_OFF_RATIO,
)
from netsim.anomalies import RollingZScoreDetector, AnomalySchedule


# ═══════════════════════════════════════════════════════════════════════════
# Simulator context — shared state injected into every tool call
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimulatorContext:
    """Mutable bag of references passed to every tool function."""
    simulator: Simulator
    detector: RollingZScoreDetector | None = None
    anomaly_schedule: AnomalySchedule | None = None
    satellite_pass: dict[str, np.ndarray] | None = None  # from simulate_pass()
    device_params: dict[int, dict[str, Any]] = field(default_factory=dict)
    # device_params[dev_id] = {"sf": ..., "tx_power_dbm": ...,
    #                          "channel": ..., "link": ...}


# ═══════════════════════════════════════════════════════════════════════════
# Validation constants
# ═══════════════════════════════════════════════════════════════════════════

VALID_SFS = {7, 8, 9, 10, 11, 12}
VALID_POWER_RANGE = (2, 14)          # dBm
VALID_CHANNELS = set(EU868_CHANNELS)
VALID_LINKS = {"terrestrial", "satellite", "wait"}


# ═══════════════════════════════════════════════════════════════════════════
# 1.  get_link_stats
# ═══════════════════════════════════════════════════════════════════════════

def get_link_stats(
    ctx: SimulatorContext,
    scope: str = "all",
    window_s: float = 300.0,
) -> dict[str, Any]:
    """Return link statistics over a recent time window.

    Parameters
    ----------
    scope : str
        ``"all"`` for network-wide stats, or a device ID (as string) for
        a single device.
    window_s : float
        Look-back window in simulated seconds.

    Returns
    -------
    dict
        mean_snr_db, min_snr_db, per, current_sf, current_power_dbm,
        link, toa_s, duty_cycle_budget_s, n_packets.
    """
    sim = ctx.simulator
    t_min = max(0.0, sim.now - window_s)

    # Filter packets in window
    if scope == "all":
        pkts = [p for p in sim.packets if p.end_time >= t_min]
    else:
        dev_id = int(scope)
        pkts = [p for p in sim.packets
                if p.device_id == dev_id and p.end_time >= t_min]

    if not pkts:
        return {
            "mean_snr_db": None,
            "min_snr_db": None,
            "per": None,
            "current_sf": None,
            "current_power_dbm": None,
            "link": None,
            "toa_s": None,
            "duty_cycle_budget_s": None,
            "n_packets": 0,
        }

    snrs = [p.snr_db for p in pkts]
    n_failed = sum(1 for p in pkts if not p.delivered)
    last = pkts[-1]

    # Duty-cycle remaining budget
    toa = time_on_air(sim.payload_bytes, last.sf)
    off_until = sim._duty_off_until.get(last.device_id, {}).get(
        last.channel, 0.0
    )
    budget = max(0.0, off_until - sim.now)

    return {
        "mean_snr_db": round(float(np.mean(snrs)), 1),
        "min_snr_db": round(float(np.min(snrs)), 1),
        "per": round(n_failed / len(pkts), 3),
        "current_sf": last.sf,
        "current_power_dbm": last.tx_power_dbm,
        "link": last.link,
        "toa_s": round(toa, 4),
        "duty_cycle_budget_s": round(budget, 2),
        "n_packets": len(pkts),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2.  get_satellite_status
# ═══════════════════════════════════════════════════════════════════════════

def get_satellite_status(
    ctx: SimulatorContext,
    device_id: int = 0,
) -> dict[str, Any]:
    """Return satellite visibility and Doppler information.

    Parameters
    ----------
    device_id : int
        Device requesting satellite status.

    Returns
    -------
    dict
        elevation_now_deg, elevation_1min_deg, elevation_5min_deg,
        elevation_10min_deg, doppler_shift_hz, doppler_rate_hz_s,
        time_to_pass_end_s, satellite_available.
    """
    sp = ctx.satellite_pass
    if sp is None or len(sp["t_s"]) == 0:
        return {
            "elevation_now_deg": None,
            "elevation_1min_deg": None,
            "elevation_5min_deg": None,
            "elevation_10min_deg": None,
            "doppler_shift_hz": None,
            "doppler_rate_hz_s": None,
            "time_to_pass_end_s": None,
            "satellite_available": False,
        }

    now = ctx.simulator.now
    t_arr = sp["t_s"]

    def _interp(offset: float) -> float | None:
        val = float(np.interp(now + offset, t_arr, sp["elevation_deg"],
                               left=0.0, right=0.0))
        return round(val, 1) if val > 0 else None

    doppler = float(np.interp(now, t_arr, sp["doppler_hz"],
                               left=0.0, right=0.0))
    doppler_rate = float(np.interp(now, t_arr, sp["doppler_rate_hz_s"],
                                    left=0.0, right=0.0))

    # Time until pass ends (elevation drops to 0 in the array)
    future_mask = t_arr > now
    if np.any(future_mask):
        time_to_end = round(float(t_arr[future_mask][-1]) - now, 1)
    else:
        time_to_end = 0.0

    elev_now = _interp(0.0)
    available = elev_now is not None and elev_now > 0

    # Check anomaly schedule
    if ctx.anomaly_schedule and not ctx.anomaly_schedule.is_satellite_available(now):
        available = False

    return {
        "elevation_now_deg": _interp(0.0),
        "elevation_1min_deg": _interp(60.0),
        "elevation_5min_deg": _interp(300.0),
        "elevation_10min_deg": _interp(600.0),
        "doppler_shift_hz": round(doppler, 0),
        "doppler_rate_hz_s": round(doppler_rate, 1),
        "time_to_pass_end_s": time_to_end,
        "satellite_available": available,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  get_anomaly_report
# ═══════════════════════════════════════════════════════════════════════════

def get_anomaly_report(
    ctx: SimulatorContext,
    scope: str = "all",
) -> dict[str, Any]:
    """Return the anomaly detector's latest report.

    Parameters
    ----------
    scope : str
        ``"all"`` for network-wide, or a channel frequency as string.

    Returns
    -------
    dict
        has_alert, suspected_type, start_time, affected_channels,
        affected_devices, confidence, n_alerts.
    """
    if ctx.detector is None:
        return {
            "has_alert": False,
            "suspected_type": None,
            "start_time": None,
            "affected_channels": [],
            "affected_devices": [],
            "confidence": 0.0,
            "n_alerts": 0,
        }

    report = ctx.detector.latest_report(
        lookback_s=300.0, now=ctx.simulator.now
    )

    # Filter by channel scope if requested
    if scope != "all" and report["has_alert"]:
        try:
            ch = float(scope)
            if ch not in report["affected_channels"]:
                report = {
                    "has_alert": False,
                    "suspected_type": None,
                    "start_time": None,
                    "affected_channels": [],
                    "affected_devices": [],
                    "confidence": 0.0,
                    "n_alerts": 0,
                }
        except ValueError:
            pass  # invalid scope, return full report

    return report


# ═══════════════════════════════════════════════════════════════════════════
# 4.  get_network_load
# ═══════════════════════════════════════════════════════════════════════════

def get_network_load(
    ctx: SimulatorContext,
) -> dict[str, Any]:
    """Return current network load: packets per SF/channel, collision rate.

    Returns
    -------
    dict
        packets_by_sf (dict[int, int]),
        packets_by_channel (dict[float, int]),
        collision_rate (float),
        total_packets (int).
    """
    sim = ctx.simulator
    window_s = 300.0
    t_min = max(0.0, sim.now - window_s)
    recent = [p for p in sim.packets if p.end_time >= t_min]

    if not recent:
        return {
            "packets_by_sf": {},
            "packets_by_channel": {},
            "collision_rate": 0.0,
            "total_packets": 0,
        }

    by_sf: dict[int, int] = {}
    by_ch: dict[float, int] = {}
    collisions = 0

    for p in recent:
        by_sf[p.sf] = by_sf.get(p.sf, 0) + 1
        by_ch[p.channel] = by_ch.get(p.channel, 0) + 1
        if p.collided:
            collisions += 1

    return {
        "packets_by_sf": by_sf,
        "packets_by_channel": by_ch,
        "collision_rate": round(collisions / len(recent), 3),
        "total_packets": len(recent),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  set_tx_params
# ═══════════════════════════════════════════════════════════════════════════

def set_tx_params(
    ctx: SimulatorContext,
    device_ids: list[int],
    sf: int,
    tx_power_dbm: int,
    channel: float,
    link: str,
) -> dict[str, str]:
    """Apply new transmission parameters for a set of devices.

    Validates against JSON-schema-like constraints before applying.
    Returns ``{"status": "applied"}`` or ``{"status": "rejected",
    "reason": "..."}`` as specified.

    Parameters
    ----------
    device_ids : list[int]
        Target device IDs.
    sf : int
        Spreading factor (7-12).
    tx_power_dbm : int
        Transmit power in dBm (2-14).
    channel : float
        Channel frequency in MHz.
    link : str
        ``"terrestrial"``, ``"satellite"``, or ``"wait"``.

    Returns
    -------
    dict
        ``{"status": "applied", "devices": [...]}`` or
        ``{"status": "rejected", "reason": "..."}``.
    """
    # --- Validation (guardrails) -------------------------------------------
    if sf not in VALID_SFS:
        return {"status": "rejected",
                "reason": f"SF must be 7-12, got {sf}"}

    if not (VALID_POWER_RANGE[0] <= tx_power_dbm <= VALID_POWER_RANGE[1]):
        return {"status": "rejected",
                "reason": f"Power must be {VALID_POWER_RANGE[0]}-"
                          f"{VALID_POWER_RANGE[1]} dBm, got {tx_power_dbm}"}

    if channel not in VALID_CHANNELS:
        return {"status": "rejected",
                "reason": f"Channel {channel} not in {sorted(VALID_CHANNELS)}"}

    if link not in VALID_LINKS:
        return {"status": "rejected",
                "reason": f"Link must be one of {sorted(VALID_LINKS)}, "
                          f"got '{link}'"}

    if not device_ids:
        return {"status": "rejected", "reason": "device_ids is empty"}

    # --- Check gateway / satellite availability ----------------------------
    if link == "terrestrial":
        if (ctx.anomaly_schedule
                and ctx.anomaly_schedule.is_gateway_down(ctx.simulator.now)):
            return {"status": "rejected",
                    "reason": "Terrestrial gateway is currently down"}

    if link == "satellite":
        if (ctx.anomaly_schedule
                and not ctx.anomaly_schedule.is_satellite_available(
                    ctx.simulator.now)):
            return {"status": "rejected",
                    "reason": "Satellite is not available (pass ended)"}

    # --- Apply to device params -------------------------------------------
    for dev_id in device_ids:
        ctx.device_params[dev_id] = {
            "sf": sf,
            "tx_power_dbm": tx_power_dbm,
            "channel": channel,
            "link": link,
        }

    return {"status": "applied", "devices": device_ids}


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Tool registry (for the agent loop)
# ═══════════════════════════════════════════════════════════════════════════

# Wrapper functions with simple signatures for the ollama tool-calling
# interface.  Each takes keyword arguments matching the LLM's tool call
# and delegates to the full function with the injected context.

def make_tool_registry(ctx: SimulatorContext) -> dict[str, callable]:
    """Build a name -> callable mapping bound to a SimulatorContext.

    The returned callables accept only the parameters the LLM provides
    (no ctx argument).
    """

    def _get_link_stats(scope: str = "all", window_s: float = 300.0):
        """Get link statistics: mean/min SNR, PER, SF, power, ToA, duty-cycle budget."""
        return get_link_stats(ctx, scope=scope, window_s=window_s)

    def _get_satellite_status(device_id: int = 0):
        """Get satellite status: elevation now and in 1/5/10 min, Doppler shift/rate, time to pass end."""
        return get_satellite_status(ctx, device_id=device_id)

    def _get_anomaly_report(scope: str = "all"):
        """Get anomaly detector report: suspected type, start time, affected channels/devices, confidence."""
        return get_anomaly_report(ctx, scope=scope)

    def _get_network_load():
        """Get network load: packets per SF and channel, collision rate."""
        return get_network_load(ctx)

    def _set_tx_params(
        device_ids: list[int],
        sf: int,
        tx_power_dbm: int,
        channel: float,
        link: str,
    ):
        """Set transmission parameters for devices. Returns 'applied' or 'rejected' with reason."""
        return set_tx_params(
            ctx,
            device_ids=device_ids,
            sf=sf,
            tx_power_dbm=tx_power_dbm,
            channel=channel,
            link=link,
        )

    return {
        "get_link_stats": _get_link_stats,
        "get_satellite_status": _get_satellite_status,
        "get_anomaly_report": _get_anomaly_report,
        "get_network_load": _get_network_load,
        "set_tx_params": _set_tx_params,
    }
