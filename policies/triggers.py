"""Event triggers shared by the LLM agent and the rule-based controller.

Section 3.5 phase 6: the controller is called only when

1. the anomaly detector raises an alert,
2. the loss rate over the last packets exceeds 10 %,
3. a satellite elevation threshold is crossed (rise, set, and a warning
   two minutes before the set so that commands can still be delivered),
4. a periodic review is due (every 30 simulated minutes).

A cooldown (10 min by default) prevents the same trigger from firing again
while an event is ongoing; without it the detector would call the agent at
every tick of a long event.  All inputs are network-server-side.
"""

from __future__ import annotations

from typing import Any

PASS_WARNING_S = 120.0


def ns_loss_rate(sim: Any, now: float, window_s: float = 600.0) -> tuple[float, int]:
    """Loss rate inferred from frame-counter gaps over the window."""
    recs = sim.ns.window(now, window_s)
    lost = sum(r.lost_before for r in recs)
    n = lost + len(recs)
    return (lost / n if n else 0.0), n


def affected_devices(sim: Any, now: float, drop_db: float = 6.0) -> list[int]:
    """Devices that look degraded from the network side.

    * silent: no uplink for more than 2.5 reporting intervals, or
    * last SNR ≥ *drop_db* below the median of their earlier uplinks.
    """
    out = []
    interval = sim.interval
    for dev in sim.devices:
        if dev.home_link != "terrestrial":
            continue
        last = sim.ns.last_seen.get(dev.device_id)
        if now > 2.5 * interval and (last is None or now - last > 2.5 * interval):
            out.append(dev.device_id)
            continue
        h = sim.ns.dev_history.get(dev.device_id, [])
        if len(h) >= 4:
            ref = sorted(r.snr_db for r in h[:-1])[len(h[:-1]) // 2]
            if h[-1].snr_db <= ref - drop_db and now - h[-1].time < 2 * interval:
                out.append(dev.device_id)
    return out


class TriggerManager:
    def __init__(self, agent_cfg: dict[str, Any]) -> None:
        self.per_trigger = float(agent_cfg["per_trigger"])
        self.review_s = float(agent_cfg["periodic_review_s"])
        self.cooldown_s = float(agent_cfg["cooldown_s"])
        self.last: dict[tuple, float] = {}
        self.next_review = self.review_s
        self.prev_visible: bool | None = None
        self.warned_pass: set[float] = set()
        self.fired: list[dict[str, Any]] = []

    def _cool(self, key: tuple, now: float) -> bool:
        t = self.last.get(key)
        if t is not None and now - t < self.cooldown_s:
            return False
        self.last[key] = now
        return True

    def check(self, sim: Any, now: float, alerts: list) -> list[dict[str, Any]]:
        reasons: list[dict[str, Any]] = []
        for a in alerts:
            if self._cool(("alert", a.kind), now):
                reasons.append({"trigger": "anomaly_alert", "kind": a.kind})
        loss, n = ns_loss_rate(sim, now)
        if n >= 20 and loss > self.per_trigger and self._cool(("loss",), now):
            reasons.append({"trigger": "high_loss", "loss_rate": round(loss, 3)})
        if sim.passes:
            st = sim.satellite_state(now)
            vis = st["visible"]
            if self.prev_visible is not None and vis != self.prev_visible:
                reasons.append({"trigger": "satellite_rise" if vis else "satellite_set"})
            self.prev_visible = vis
            if vis and st["time_to_set_s"] <= PASS_WARNING_S:
                p = st["pass"]
                if p.t_set not in self.warned_pass:
                    self.warned_pass.add(p.t_set)
                    reasons.append({"trigger": "satellite_set_soon",
                                    "time_to_set_s": round(st["time_to_set_s"])})
        if now >= self.next_review:
            self.next_review = now + self.review_s
            reasons.append({"trigger": "periodic_review"})
        if reasons:
            self.fired.append({"t": now, "reasons": reasons})
        return reasons
