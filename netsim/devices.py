"""End devices: placement, slow channel state, keyed random streams.

Fairness between policies relies on *common random numbers*: every random
quantity that does not depend on the policy (positions, shadowing, reading
times, per-packet fading, LOS/NLOS state, the uniform used for the PER
Bernoulli draw, the channel-hopping draw) is pre-drawn per device from a
stream keyed by (seed, device id).  Two policies run with the same seed see
exactly the same channel realisations; only their decisions differ, so
policy comparisons are paired.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from channel.link_budget import hata_rural_loss, hata_urban_loss
from phy.css import SNR_THRESHOLDS_DB


@dataclass(frozen=True)
class TxParams:
    """Transmission parameters applied by a device."""
    sf: int
    tx_power_dbm: int
    channels: tuple[float, ...]
    link: str                      # "terrestrial" | "satellite" | "wait"

    def replace(self, **kw: Any) -> "TxParams":
        d = {"sf": self.sf, "tx_power_dbm": self.tx_power_dbm,
             "channels": self.channels, "link": self.link}
        d.update({k: v for k, v in kw.items() if v is not None})
        if isinstance(d["channels"], list):
            d["channels"] = tuple(d["channels"])
        return TxParams(**d)


@dataclass
class Device:
    device_id: int
    x_km: float
    y_km: float
    distance_km: float
    remote: bool                    # placed outside terrestrial coverage
    terr_gain_db: float             # G_tx + G_rx − L_Hata − shadowing (slow)
    home_link: str                  # link assigned at deployment
    reading_times: np.ndarray       # policy-independent reading schedule
    # per-reading random numbers (index = reading number)
    u_per: np.ndarray
    z_fast: np.ndarray
    u_los: np.ndarray
    z_sat: np.ndarray
    u_channel: np.ndarray
    # mutable state
    base: TxParams | None = None    # set by the base controller (fixed/ADR/ML)
    override: dict[str, Any] | None = None
    override_until: float = -1.0
    next_tx_allowed: float = 0.0    # duty-cycle bookkeeping (sub-band g1)
    fcnt: int = 0
    stats: dict[str, float] = field(default_factory=lambda: {
        "readings": 0, "sent": 0, "delivered": 0, "withheld": 0,
        "dropped_duty": 0, "energy_j": 0.0})

    def effective_params(self, t: float) -> TxParams:
        """Base parameters with the active override (if any) applied."""
        if self.override is not None and t < self.override_until:
            return self.base.replace(**self.override)
        return self.base


def terrestrial_noise_floor_dbm(cfg: dict[str, Any]) -> float:
    bw = cfg["phy"]["bandwidth_hz"]
    return -174.0 + 10.0 * math.log10(bw) + cfg["terrestrial"]["noise_figure_db"]


def create_devices(cfg: dict[str, Any], seed: int) -> list[Device]:
    """Place devices and pre-draw every policy-independent random number."""
    net, dep, ter = cfg["network"], cfg["deployment"], cfg["terrestrial"]
    n = int(net["n_devices"])
    duration = float(cfg["duration_s"])
    interval = float(net["report_interval_s"])
    jitter = float(net["report_jitter_frac"])
    surges = [a for a in cfg.get("anomalies", []) if a["type"] == "traffic_surge"]
    noise_floor = terrestrial_noise_floor_dbm(cfg)
    hata = hata_rural_loss if ter["environment"] == "rural" else hata_urban_loss
    n_remote = int(round(n * float(dep.get("remote_fraction", 0.0))))
    devices: list[Device] = []
    for i in range(n):
        rng = np.random.default_rng([int(seed), 7919, i])
        remote = i >= n - n_remote
        if remote:
            r0, r1 = dep["remote_radius_km"]
        else:
            r0, r1 = 0.1, float(dep["core_radius_km"])
        r = math.sqrt(rng.uniform(r0 ** 2, r1 ** 2))
        th = rng.uniform(0.0, 2.0 * math.pi)
        d_hata = max(r, 1.0)                      # Hata is defined from 1 km
        loss = hata(ter["frequency_mhz"], ter["gw_height_m"], ter["device_height_m"], d_hata)
        shadow = rng.normal(0.0, ter["shadowing_std_db"])
        gain = ter["device_antenna_gain_dbi"] + ter["gw_antenna_gain_dbi"] - loss - shadow
        median_snr_max = 14.0 + gain - noise_floor
        terr_ok = median_snr_max >= SNR_THRESHOLDS_DB[12] + ter["coverage_margin_db"]
        sat_enabled = cfg["satellite"].get("enabled", True)
        home = "terrestrial" if (terr_ok or not sat_enabled) else "satellite"
        # regular reading schedule: period ~ U(1−j, 1+j) × interval, random phase
        times = []
        t = rng.uniform(0.0, interval)
        while t < duration:
            times.append(t)
            t += interval * rng.uniform(1 - jitter, 1 + jitter)
        # traffic surge: extra event-driven readings (Poisson) on top of the
        # regular schedule, so that the total rate is rate_multiplier × normal.
        # (Speeding up the periodic timer instead would leave every device in
        # phase when the surge ends: an artefact, not a property of the event.)
        for a in surges:
            extra_rate = (float(a["rate_multiplier"]) - 1.0) / interval
            t = float(a["start_s"]) + rng.exponential(1.0 / extra_rate)
            while t < float(a["start_s"]) + float(a["duration_s"]) and t < duration:
                times.append(t)
                t += rng.exponential(1.0 / extra_rate)
        times.sort()
        k = len(times) + 1          # one set of random numbers per reading
        devices.append(Device(
            device_id=i, x_km=r * math.cos(th), y_km=r * math.sin(th), distance_km=r,
            remote=remote, terr_gain_db=gain, home_link=home,
            reading_times=np.asarray(times),
            u_per=rng.random(k), z_fast=rng.standard_normal(k), u_los=rng.random(k),
            z_sat=rng.standard_normal(k), u_channel=rng.random(k),
        ))
    return devices


def warm_snr_history(dev: Device, cfg: dict[str, Any], seed: int, n: int = 20,
                     tx_power_dbm: float = 14.0) -> list[float]:
    """SNR samples of *n* past uplinks on the home terrestrial link (warm start).

    Drawn from a stream separate from the evaluation stream.
    """
    rng = np.random.default_rng([int(seed), 104729, dev.device_id])
    nf = terrestrial_noise_floor_dbm(cfg)
    std = cfg["terrestrial"]["fast_fading_std_db"]
    return list(tx_power_dbm + dev.terr_gain_db - nf + std * rng.standard_normal(n))
