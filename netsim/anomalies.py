"""Anomaly injection (ground truth) and a rolling z-score detector.

Section 3.5, phase 5.  Five event types:

| Event              | Injection                                               |
|--------------------|---------------------------------------------------------|
| jamming            | interference I dB above the noise floor on one channel  |
|                    | at the terrestrial gateway: SNR − 10·log10(1 + 10^(I/10))|
| deep_fade          | extra loss for a subset of devices (both links)         |
| satellite_pass_end | the pass ends (elevation below the minimum)             |
| gateway_outage     | terrestrial gateway receives nothing                    |
| traffic_surge      | reading rate multiplied (applied in netsim.devices)     |

The detector only sees what a network server sees (gateway noise-floor
measurements, delivered uplinks, losses inferred from frame-counter gaps),
never the ground truth.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Ground truth
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnomalyEvent:
    type: str
    start_s: float
    end_s: float
    params: dict[str, Any] = field(default_factory=dict)

    def active(self, t: float) -> bool:
        return self.start_s <= t < self.end_s


class AnomalySchedule:
    """Ground-truth events of one run (built from the config)."""

    def __init__(self, events: list[AnomalyEvent] | None = None) -> None:
        self.events = list(events or [])

    @classmethod
    def from_config(cls, cfg: dict[str, Any], terrestrial_ids: list[int], seed: int,
                    pass_end_times: list[float] | None = None) -> "AnomalySchedule":
        events = []
        for a in cfg.get("anomalies", []):
            typ = a["type"]
            start = float(a["start_s"])
            if typ == "satellite_pass_end":
                # The event *is* the end of the pass; it lasts until the run ends.
                end = float(cfg["duration_s"])
                events.append(AnomalyEvent(typ, start, end, {}))
                continue
            end = start + float(a["duration_s"])
            params = {k: v for k, v in a.items() if k not in ("type", "start_s", "duration_s")}
            if typ == "deep_fade":
                rng = np.random.default_rng([int(seed), 31337])
                k = int(round(len(terrestrial_ids) * float(a["device_fraction"])))
                params["device_ids"] = set(int(x) for x in rng.choice(terrestrial_ids, size=k, replace=False))
            events.append(AnomalyEvent(typ, start, end, params))
        return cls(events)

    def active(self, t: float) -> list[AnomalyEvent]:
        return [e for e in self.events if e.active(t)]

    def interference_db(self, t: float, channel: float) -> float:
        """Jammer power above the noise floor (dB) on *channel*, or −inf."""
        out = -math.inf
        for e in self.events:
            if e.type == "jamming" and e.active(t) and abs(e.params["channel_mhz"] - channel) < 1e-6:
                out = max(out, float(e.params["interference_db"]))
        return out

    def snr_penalty_db(self, t: float, channel: float) -> float:
        """SNR loss at the terrestrial gateway due to jamming: 10·log10(1 + I/N)."""
        i = self.interference_db(t, channel)
        return 0.0 if i == -math.inf else 10.0 * math.log10(1.0 + 10.0 ** (i / 10.0))

    def extra_loss_db(self, t: float, device_id: int) -> float:
        loss = 0.0
        for e in self.events:
            if e.type == "deep_fade" and e.active(t) and device_id in e.params["device_ids"]:
                loss += float(e.params["extra_loss_db"])
        return loss

    def gateway_down(self, t: float) -> bool:
        return any(e.type == "gateway_outage" and e.active(t) for e in self.events)

    def label(self, t: float) -> str | None:
        act = self.active(t)
        return act[0].type if act else None


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Detector
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DetectorAlert:
    time: float
    kind: str                  # noise_rise | snr_drop | loss_spike | load_rise | gateway_silence | device_silence
    suspected_type: str        # jamming | deep_fade | gateway_outage | traffic_surge | unknown
    channel: float | None
    device_ids: list[int]
    z_score: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {"time_s": round(self.time, 1), "kind": self.kind,
                "suspected_type": self.suspected_type, "channel_mhz": self.channel,
                "n_devices": len(self.device_ids), "z": round(self.z_score, 2),
                "confidence": round(self.confidence, 2)}


class RollingZScoreDetector:
    """Rolling z-score detector on network-server statistics.

    Statistics (all network-server side):

    * per-channel noise floor at the terrestrial gateway, every tick (60 s);
    * per-channel mean SNR of delivered uplinks, per 5-min block;
    * network loss rate inferred from frame-counter gaps, per 5-min block
      (losses cannot be attributed to a channel: the gap does not say on
      which channel the missing frames were sent);
    * uplink count (received + inferred lost), per 5-min block;
    * fraction of silent terrestrial devices, per 5-min block;
    * gateway silence (no uplink at all for ``silence_s``).

    z-scores: rolling mean/std of the last *window* normal samples for the
    noise floor, SNR and silent fraction; a binomial test against the rolling
    mean for the loss rate; a Poisson test for the count.  An alert needs
    *persistence* consecutive exceedances and is emitted once, on the rising
    edge; samples taken while in alert are not added to the reference, so a
    long event does not become the new normal.

    The first version computed a per-channel loss rate from frame-counter
    gaps over a sliding window: autocorrelated, wrongly attributed, and it
    produced hundreds of false alerts per 3-h run.
    """

    def __init__(self, z_threshold: float = 3.0, window: int = 30,
                 min_samples: int = 10, silence_s: float = 300.0,
                 persistence: int = 2) -> None:
        self.z = z_threshold
        self.window = window
        self.min_samples = min_samples
        self.silence_s = silence_s
        self.persistence = persistence
        self._ref: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self._streak: dict[tuple, int] = defaultdict(int)
        self.active: dict[tuple, float] = {}
        self.alerts: list[DetectorAlert] = []

    def _z(self, key: tuple, value: float, kind: str, n: float = 0.0) -> float | None:
        ref = self._ref[key]
        if len(ref) < self.min_samples:
            return None
        arr = np.asarray(ref, dtype=float)
        mu = float(arr.mean())
        if kind == "binomial":
            p = min(max(mu, 0.005), 0.995)
            return (value - mu) / math.sqrt(p * (1 - p) / max(n, 1.0))
        if kind == "poisson":
            return (value - mu) / math.sqrt(max(mu, 1.0))
        return (value - mu) / max(float(arr.std(ddof=1)), 1e-3)

    def _update(self, t: float, key: tuple, value: float | None, direction: int,
                kind: str = "gauss", n: float = 0.0) -> float | None:
        """Return z on the rising edge of an alert, else None."""
        if value is None or not np.isfinite(value):
            return None
        z = self._z(key, value, kind, n)
        exceed = z is not None and direction * z > self.z
        if exceed:
            self._streak[key] += 1
            if self._streak[key] >= self.persistence and key not in self.active:
                self.active[key] = t
                return z
            return None
        self._streak[key] = 0
        self.active.pop(key, None)
        self._ref[key].append(value)
        return None

    def observe(self, t: float, obs: dict[str, Any]) -> list[DetectorAlert]:
        new: list[DetectorAlert] = []
        for ch, v in obs.get("noise_floor_dbm", {}).items():
            z = self._update(t, ("noise", ch), v, +1)
            if z is not None:
                new.append(DetectorAlert(t, "noise_rise", "jamming", ch, [], z, z / self.z))
        blk = obs.get("block")
        if blk is not None:
            for ch, (v, n) in blk.get("mean_snr", {}).items():
                if n >= 5:
                    z = self._update(t, ("snr", ch), v, -1)
                    if z is not None:
                        new.append(DetectorAlert(t, "snr_drop", "deep_fade", ch,
                                                 list(blk.get("affected", [])), z, abs(z) / self.z))
            lost, n = blk.get("loss", (0, 0))
            if n >= 20:
                z = self._update(t, ("loss",), lost / n, +1, "binomial", n)
                if z is not None:
                    new.append(DetectorAlert(t, "loss_spike", "unknown", None,
                                             list(blk.get("affected", [])), z, z / self.z))
            z = self._update(t, ("load",), blk.get("uplinks"), +1, "poisson")
            if z is not None:
                new.append(DetectorAlert(t, "load_rise", "traffic_surge", None, [], z, z / self.z))
            z = self._update(t, ("silent",), blk.get("silent_fraction"), +1)
            if z is not None and blk.get("silent_fraction", 0.0) >= 0.03:
                new.append(DetectorAlert(t, "device_silence", "deep_fade", None,
                                         list(blk.get("silent", [])), z, z / self.z))
        silent_gw = obs.get("gw_silent_s", 0.0) >= self.silence_s
        if silent_gw and ("gw",) not in self.active:
            self.active[("gw",)] = t
            new.append(DetectorAlert(t, "gateway_silence", "gateway_outage", None, [], 0.0, 1.0))
        elif not silent_gw:
            self.active.pop(("gw",), None)
        self.alerts.extend(new)
        return new

    def active_kinds(self) -> list[str]:
        names = {"noise": "noise_rise", "snr": "snr_drop", "loss": "loss_spike",
                 "load": "load_rise", "silent": "device_silence", "gw": "gateway_silence"}
        return sorted({names[k[0]] for k in self.active})

    def report(self, now: float, lookback_s: float = 600.0) -> dict[str, Any]:
        """Summary of recent alerts (used by the agent tool and the triggers)."""
        recent = [a for a in self.alerts if a.time >= now - lookback_s]
        if not recent and not self.active:
            return {"has_alert": False, "suspected_type": None, "start_time_s": None,
                    "affected_channels_mhz": [], "affected_device_ids": [],
                    "n_affected_devices": 0, "confidence": 0.0,
                    "ongoing_conditions": [], "alerts": []}
        if not recent:      # condition still active, alert raised earlier
            recent = [a for a in self.alerts if a.time >= min(self.active.values())]
        counts: dict[str, int] = defaultdict(int)
        for a in recent:
            counts[a.suspected_type] += 1
        known = {k: v for k, v in counts.items() if k != "unknown"}
        top = max(known or counts, key=(known or counts).get)
        devices = sorted({d for a in recent for d in a.device_ids})
        return {
            "has_alert": True,
            "suspected_type": top,
            "start_time_s": round(min(a.time for a in recent), 1),
            "affected_channels_mhz": sorted({a.channel for a in recent if a.channel is not None}),
            "affected_device_ids": devices[:50],
            "n_affected_devices": len(devices),
            "confidence": round(max(a.confidence for a in recent), 2),
            "ongoing_conditions": self.active_kinds(),
            "alerts": [a.as_dict() for a in recent[-6:]],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Precision / recall against the injected events
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_detector(alerts: list[DetectorAlert], schedule: AnomalySchedule,
                      tolerance_s: float = 600.0) -> dict[str, float]:
    """Alert-level precision and event-level recall.

    * An alert is a true positive if it falls in [start, end + tolerance] of
      some injected event (the tolerance covers detection delay at the end).
    * An event is detected if at least one alert falls in
      [start, end + tolerance].
    * Detection delay: first matching alert time − event start.
    """
    # The end of a satellite pass is predictable from the ephemeris and is
    # handled by the elevation trigger, not by the detector: not scored here.
    events = [e for e in schedule.events if e.type != "satellite_pass_end"]
    tp = 0
    for a in alerts:
        if any(e.start_s <= a.time <= e.end_s + tolerance_s for e in events):
            tp += 1
    fp = len(alerts) - tp
    detected, delays = 0, []
    for e in events:
        hits = [a.time for a in alerts if e.start_s <= a.time <= e.end_s + tolerance_s]
        if hits:
            detected += 1
            delays.append(min(hits) - e.start_s)
    n_ev = len(events)
    precision = tp / len(alerts) if alerts else float("nan")
    recall = detected / n_ev if n_ev else float("nan")
    return {"alerts": len(alerts), "true_alerts": tp, "false_alerts": fp,
            "events": n_ev, "events_detected": detected,
            "precision": precision, "recall": recall,
            "detection_delay_s": float(np.mean(delays)) if delays else float("nan")}
