"""Anomaly injection and detection for the LoRa network simulator.

Section 3.5 Phase 5 — five anomaly types and a rolling z-score detector.

Anomaly types
-------------
| Event              | How to simulate                                       |
|--------------------|-------------------------------------------------------|
| Jamming burst      | +10 to +20 dB interference on one channel, 2-10 min   |
| Deep fade/blockage | +15 to +25 dB extra loss for a device subset, ~mins   |
| End of sat. pass   | Elevation falls below minimum                         |
| Gateway outage     | Terrestrial gateway down for 30 min                   |
| Traffic surge      | 5x the normal reporting rate                          |

Detector: rolling z-score on each channel's noise floor, SNR, and PER.
Report precision and recall against the injected ground-truth labels.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Anomaly event definitions
# ═══════════════════════════════════════════════════════════════════════════

class AnomalyType(Enum):
    JAMMING = auto()
    DEEP_FADE = auto()
    SATELLITE_PASS_END = auto()
    GATEWAY_OUTAGE = auto()
    TRAFFIC_SURGE = auto()


@dataclass
class AnomalyEvent:
    """Ground-truth record of an injected anomaly."""
    anomaly_type: AnomalyType
    start_time: float            # simulation seconds
    end_time: float              # simulation seconds
    params: dict[str, Any] = field(default_factory=dict)
    # params examples:
    #   JAMMING:       {"channel": 868.1, "interference_db": 15.0}
    #   DEEP_FADE:     {"device_ids": [0,1,2], "extra_loss_db": 20.0}
    #   SAT_PASS_END:  {"min_elevation_deg": 10.0}
    #   GW_OUTAGE:     {}
    #   TRAFFIC_SURGE: {"rate_multiplier": 5.0}

    def is_active(self, t: float) -> bool:
        return self.start_time <= t < self.end_time


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Anomaly schedule — injection engine
# ═══════════════════════════════════════════════════════════════════════════

class AnomalySchedule:
    """Container for all anomaly events in a simulation run.

    Provides query methods that the simulator calls to modify SNR,
    block gateways, or scale traffic at each time step.
    """

    def __init__(self) -> None:
        self.events: list[AnomalyEvent] = []

    # ── factory helpers (one per anomaly type) ────────────────────────

    def add_jamming(
        self,
        start: float,
        duration_s: float,
        channel: float,
        interference_db: float,
    ) -> AnomalyEvent:
        """Jamming burst: +10 to +20 dB interference on *channel*.

        Parameters
        ----------
        start : float
            Start time (s).
        duration_s : float
            Duration in seconds (spec: 2–10 minutes = 120–600 s).
        channel : float
            Affected channel (MHz).
        interference_db : float
            Interference power above the noise floor (dB), range 10–20.
        """
        ev = AnomalyEvent(
            anomaly_type=AnomalyType.JAMMING,
            start_time=start,
            end_time=start + duration_s,
            params={"channel": channel, "interference_db": interference_db},
        )
        self.events.append(ev)
        return ev

    def add_deep_fade(
        self,
        start: float,
        duration_s: float,
        device_ids: list[int],
        extra_loss_db: float,
    ) -> AnomalyEvent:
        """Deep fade / blockage: +15 to +25 dB extra loss for a subset of devices.

        Parameters
        ----------
        start : float
            Start time (s).
        duration_s : float
            Duration in seconds.
        device_ids : list[int]
            Affected devices.
        extra_loss_db : float
            Additional path loss (dB), range 15–25.
        """
        ev = AnomalyEvent(
            anomaly_type=AnomalyType.DEEP_FADE,
            start_time=start,
            end_time=start + duration_s,
            params={
                "device_ids": set(device_ids),
                "extra_loss_db": extra_loss_db,
            },
        )
        self.events.append(ev)
        return ev

    def add_satellite_pass_end(
        self,
        start: float,
        min_elevation_deg: float = 10.0,
    ) -> AnomalyEvent:
        """End of satellite pass: elevation falls below minimum.

        Modelled as a permanent event from *start* onward (until the next
        pass, which is outside the current simulation window).
        """
        ev = AnomalyEvent(
            anomaly_type=AnomalyType.SATELLITE_PASS_END,
            start_time=start,
            end_time=float("inf"),
            params={"min_elevation_deg": min_elevation_deg},
        )
        self.events.append(ev)
        return ev

    def add_gateway_outage(
        self,
        start: float,
        duration_s: float = 1800.0,
    ) -> AnomalyEvent:
        """Terrestrial gateway down for *duration_s* seconds (default 30 min)."""
        ev = AnomalyEvent(
            anomaly_type=AnomalyType.GATEWAY_OUTAGE,
            start_time=start,
            end_time=start + duration_s,
            params={},
        )
        self.events.append(ev)
        return ev

    def add_traffic_surge(
        self,
        start: float,
        duration_s: float,
        rate_multiplier: float = 5.0,
    ) -> AnomalyEvent:
        """Traffic surge: devices report at *rate_multiplier* x normal rate."""
        ev = AnomalyEvent(
            anomaly_type=AnomalyType.TRAFFIC_SURGE,
            start_time=start,
            end_time=start + duration_s,
            params={"rate_multiplier": rate_multiplier},
        )
        self.events.append(ev)
        return ev

    # ── query interface (called by the simulator) ─────────────────────

    def active_events(self, t: float) -> list[AnomalyEvent]:
        """Return all anomaly events active at time *t*."""
        return [ev for ev in self.events if ev.is_active(t)]

    def snr_penalty_db(
        self, t: float, device_id: int, channel: float
    ) -> float:
        """Total SNR penalty (dB, positive = worse) at time *t*.

        Combines jamming on the packet's channel and deep-fade on the
        device.
        """
        penalty = 0.0
        for ev in self.active_events(t):
            if (ev.anomaly_type == AnomalyType.JAMMING
                    and ev.params["channel"] == channel):
                penalty += ev.params["interference_db"]
            if (ev.anomaly_type == AnomalyType.DEEP_FADE
                    and device_id in ev.params["device_ids"]):
                penalty += ev.params["extra_loss_db"]
        return penalty

    def is_gateway_down(self, t: float) -> bool:
        """True if the terrestrial gateway is offline at time *t*."""
        return any(
            ev.anomaly_type == AnomalyType.GATEWAY_OUTAGE and ev.is_active(t)
            for ev in self.events
        )

    def is_satellite_available(self, t: float) -> bool:
        """True if no satellite-pass-end event is active at *t*."""
        return not any(
            ev.anomaly_type == AnomalyType.SATELLITE_PASS_END
            and ev.is_active(t)
            for ev in self.events
        )

    def traffic_rate_multiplier(self, t: float) -> float:
        """Current traffic rate multiplier (1.0 = normal)."""
        mult = 1.0
        for ev in self.active_events(t):
            if ev.anomaly_type == AnomalyType.TRAFFIC_SURGE:
                mult = max(mult, ev.params["rate_multiplier"])
        return mult

    def ground_truth_label(self, t: float) -> AnomalyType | None:
        """Return the dominant anomaly type at *t*, or None."""
        active = self.active_events(t)
        if not active:
            return None
        # Priority order (most impactful first)
        priority = [
            AnomalyType.GATEWAY_OUTAGE,
            AnomalyType.JAMMING,
            AnomalyType.DEEP_FADE,
            AnomalyType.SATELLITE_PASS_END,
            AnomalyType.TRAFFIC_SURGE,
        ]
        for atype in priority:
            if any(ev.anomaly_type == atype for ev in active):
                return atype
        return active[0].anomaly_type


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Rolling z-score anomaly detector
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _ChannelStats:
    """Rolling statistics for one channel."""
    snr_window: deque = field(default_factory=lambda: deque(maxlen=50))
    per_window: deque = field(default_factory=lambda: deque(maxlen=50))
    noise_floor_window: deque = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class DetectorAlert:
    """A single alert raised by the anomaly detector."""
    time: float
    channel: float | None
    device_ids: list[int] | None
    suspected_type: str          # human-readable label
    confidence: float            # abs(z-score) / threshold
    metric: str                  # which metric triggered ("snr", "per", "noise")
    z_score: float


class RollingZScoreDetector:
    """Anomaly detector based on rolling z-scores.

    For each channel, maintains sliding windows of SNR, PER, and noise-
    floor observations.  An alert fires when the z-score of the latest
    observation exceeds *threshold* standard deviations from the rolling
    mean.

    Section 3.5: "rolling z-score on each channel's noise floor, SNR and
    PER [...] Report its precision and recall against the injected labels."
    """

    def __init__(
        self,
        threshold: float = 3.0,
        window_size: int = 50,
        min_samples: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        threshold : float
            Number of standard deviations for an alert.
        window_size : int
            Sliding-window length for rolling mean / std.
        min_samples : int
            Minimum observations before the detector can fire.
        """
        self.threshold = threshold
        self.window_size = window_size
        self.min_samples = min_samples

        self._channels: dict[float, _ChannelStats] = defaultdict(
            lambda: _ChannelStats(
                snr_window=deque(maxlen=window_size),
                per_window=deque(maxlen=window_size),
                noise_floor_window=deque(maxlen=window_size),
            )
        )

        # Alert log (for precision / recall computation)
        self.alerts: list[DetectorAlert] = []

    def _z_score(self, window: deque, value: float) -> float | None:
        """Compute z-score of *value* relative to *window*.

        Returns None if not enough samples or std ≈ 0.
        """
        if len(window) < self.min_samples:
            return None
        arr = np.array(window, dtype=np.float64)
        mu = arr.mean()
        sigma = arr.std(ddof=1)
        if sigma < 1e-12:
            return None
        return (value - mu) / sigma

    def observe(
        self,
        time: float,
        channel: float,
        snr_db: float,
        per: float,
        noise_floor_db: float = 0.0,
        device_ids: list[int] | None = None,
    ) -> list[DetectorAlert]:
        """Feed one observation and return any new alerts.

        Call this after each packet (or batch of packets).

        Parameters
        ----------
        time : float
            Simulation time (s).
        channel : float
            Channel (MHz).
        snr_db : float
            Measured SNR (dB).
        per : float
            Recent packet error rate on this channel (0–1).
        noise_floor_db : float
            Estimated noise floor (dB), 0 if not measured.
        device_ids : list[int] or None
            Devices involved (for the alert report).

        Returns
        -------
        list[DetectorAlert]
            Alerts raised by this observation (may be empty).
        """
        stats = self._channels[channel]
        new_alerts: list[DetectorAlert] = []

        # --- SNR drop --------------------------------------------------------
        z_snr = self._z_score(stats.snr_window, snr_db)
        if z_snr is not None and z_snr < -self.threshold:
            # SNR dropped significantly → suspect jamming or deep fade
            alert = DetectorAlert(
                time=time,
                channel=channel,
                device_ids=device_ids,
                suspected_type="jamming_or_fade",
                confidence=abs(z_snr) / self.threshold,
                metric="snr",
                z_score=z_snr,
            )
            new_alerts.append(alert)

        # --- PER spike --------------------------------------------------------
        z_per = self._z_score(stats.per_window, per)
        if z_per is not None and z_per > self.threshold:
            alert = DetectorAlert(
                time=time,
                channel=channel,
                device_ids=device_ids,
                suspected_type="high_per",
                confidence=abs(z_per) / self.threshold,
                metric="per",
                z_score=z_per,
            )
            new_alerts.append(alert)

        # --- Noise-floor rise -------------------------------------------------
        if noise_floor_db != 0.0:
            z_nf = self._z_score(stats.noise_floor_window, noise_floor_db)
            if z_nf is not None and z_nf > self.threshold:
                alert = DetectorAlert(
                    time=time,
                    channel=channel,
                    device_ids=device_ids,
                    suspected_type="jamming",
                    confidence=abs(z_nf) / self.threshold,
                    metric="noise",
                    z_score=z_nf,
                )
                new_alerts.append(alert)

        # Update windows *after* computing z-scores (so the new value
        # doesn't pollute the reference distribution).
        stats.snr_window.append(snr_db)
        stats.per_window.append(per)
        if noise_floor_db != 0.0:
            stats.noise_floor_window.append(noise_floor_db)

        self.alerts.extend(new_alerts)
        return new_alerts

    def reset(self) -> None:
        """Clear all state and alert history."""
        self._channels.clear()
        self.alerts.clear()

    # ── report for the LLM agent ──────────────────────────────────────

    def latest_report(
        self, lookback_s: float = 300.0, now: float | None = None
    ) -> dict[str, Any]:
        """Build a summary report of recent alerts (for the LLM agent tool).

        Parameters
        ----------
        lookback_s : float
            How far back to look (seconds).
        now : float or None
            Current simulation time.  If None, use the latest alert time.

        Returns
        -------
        dict with keys: "has_alert", "suspected_type", "start_time",
            "affected_channels", "affected_devices", "confidence",
            "n_alerts".
        """
        if now is None and self.alerts:
            now = self.alerts[-1].time
        elif now is None:
            now = 0.0

        recent = [a for a in self.alerts if a.time >= now - lookback_s]
        if not recent:
            return {
                "has_alert": False,
                "suspected_type": None,
                "start_time": None,
                "affected_channels": [],
                "affected_devices": [],
                "confidence": 0.0,
                "n_alerts": 0,
            }

        channels = list({a.channel for a in recent if a.channel is not None})
        devices: set[int] = set()
        for a in recent:
            if a.device_ids:
                devices.update(a.device_ids)

        # Most common suspected type
        type_counts: dict[str, int] = defaultdict(int)
        for a in recent:
            type_counts[a.suspected_type] += 1
        top_type = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]

        return {
            "has_alert": True,
            "suspected_type": top_type,
            "start_time": recent[0].time,
            "affected_channels": channels,
            "affected_devices": sorted(devices),
            "confidence": max(a.confidence for a in recent),
            "n_alerts": len(recent),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Precision / Recall evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_detector(
    alerts: list[DetectorAlert],
    schedule: AnomalySchedule,
    time_tolerance_s: float = 60.0,
) -> dict[str, float]:
    """Compute precision and recall of the detector against ground truth.

    A detection is a **true positive** if any injected anomaly event is
    active within *time_tolerance_s* of the alert.  A **false negative**
    is an injected event that has no matching alert within its active
    window (extended by *time_tolerance_s*).

    Parameters
    ----------
    alerts : list[DetectorAlert]
        Alerts raised by the detector.
    schedule : AnomalySchedule
        Ground-truth injected events.
    time_tolerance_s : float
        Temporal matching tolerance.

    Returns
    -------
    dict with keys: "true_positives", "false_positives",
        "false_negatives", "precision", "recall", "f1".
    """
    # --- True positives and false positives --------------------------------
    tp = 0
    fp = 0
    matched_events: set[int] = set()  # indices of matched ground-truth events

    for alert in alerts:
        is_tp = False
        for idx, ev in enumerate(schedule.events):
            if (ev.start_time - time_tolerance_s
                    <= alert.time
                    <= ev.end_time + time_tolerance_s):
                is_tp = True
                matched_events.add(idx)
                break
        if is_tp:
            tp += 1
        else:
            fp += 1

    # --- False negatives: injected events never detected -------------------
    fn = len(schedule.events) - len(matched_events)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
