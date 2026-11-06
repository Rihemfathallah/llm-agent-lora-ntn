"""Rule-based controller on the same triggers as the LLM agent ("Triggered-Rules").

This baseline is not in the original specification.  It answers the
question an examiner will ask: *is the gain of the agent due to the LLM's
reasoning, or only to the event-triggered architecture (extra telemetry,
triggers, overrides with a lease)?*  It runs ADR by default, reacts to the
same triggers with the same network-side information, and acts through the
same override mechanism (class-A / satellite-broadcast delivery, lease).

Rules (one per anomaly family, written by hand before seeing any result):

* noise rise on channel c (jamming)      → terrestrial devices avoid c, 30 min
* SNR drop / silent devices (deep fade)  → affected devices: SF + 3 (≤ 12),
                                           14 dBm, 30 min
* gateway silence (outage)               → terrestrial devices switch to the
                                           satellite at SF10 / 14 dBm for the
                                           next pass (delivered by broadcast)
* satellite set in < 2 min               → satellite devices wait until the
                                           next pass (or the end of the run)
* satellite rise                         → satellite devices: SF10 / 14 dBm
                                           for the pass
* load rise (surge)                      → half of the SF7 devices with a
                                           ≥ 5 dB margin at SF8 move to SF8
"""

from __future__ import annotations

from typing import Any

from phy.css import SNR_THRESHOLDS_DB
from policies.adr import ADRController
from policies.triggers import TriggerManager, affected_devices


class RulesController(ADRController):
    name = "Triggered-Rules"

    def __init__(self) -> None:
        super().__init__()
        self.actions: list[dict[str, Any]] = []
        self.n_invocations = 0

    def attach(self, sim: Any) -> None:
        super().attach(sim)
        self.trig = TriggerManager(sim.cfg["agent"])

    def _issue(self, sim: Any, ids: list[int], fields: dict[str, Any], lease: float, why: str) -> None:
        if ids:
            sim.issue_override(ids, fields, lease)
            self.actions.append({"t": sim.now, "rule": why, "n": len(ids), "fields": fields,
                                 "lease_s": lease})

    def on_tick(self, sim: Any, now: float, alerts: list) -> None:
        reasons = self.trig.check(sim, now, alerts)
        if not reasons:
            return
        self.n_invocations += 1
        kinds = {r.get("kind") for r in reasons if r["trigger"] == "anomaly_alert"}
        trig = {r["trigger"] for r in reasons}
        terr = [d.device_id for d in sim.devices if d.home_link == "terrestrial"]
        sat_ids = [d.device_id for d in sim.devices if d.home_link == "satellite"]
        lease = float(sim.cfg["agent"]["default_lease_s"])

        if "noise_rise" in kinds:
            jammed = [a.channel for a in alerts if a.kind == "noise_rise"]
            ok = tuple(c for c in sim.channels if c not in jammed)
            if ok:
                self._issue(sim, terr, {"channels": ok}, lease, "avoid_jammed_channel")

        if kinds & {"snr_drop", "device_silence"}:
            ids = affected_devices(sim, now)
            for i in ids:
                dev = sim.devices[i]
                self._issue(sim, [i], {"sf": min(12, dev.base.sf + 3), "tx_power_dbm": 14},
                            lease, "boost_affected")

        if "gateway_silence" in kinds and sim.passes:
            st = sim.satellite_state(now)
            nxt = [p for p in sim.passes if p.t_rise > now]
            if st["visible"]:
                dur = st["time_to_set_s"]
            elif nxt:
                dur = 2 * nxt[0].half_duration      # lease starts at broadcast delivery
            else:
                dur = 0.0
            if dur > 0:
                self._issue(sim, terr, {"link": "satellite", "sf": 10, "tx_power_dbm": 14},
                            dur, "outage_to_satellite")

        if "satellite_set_soon" in trig and sat_ids:
            nxt = [p for p in sim.passes if p.t_rise > now]
            wait = (nxt[0].t_rise - now) if nxt else (sim.duration - now)
            self._issue(sim, sat_ids, {"link": "wait"}, wait, "wait_for_next_pass")

        if "satellite_rise" in trig and sat_ids:
            st = sim.satellite_state(now)
            if st["visible"]:
                self._issue(sim, sat_ids, {"link": "satellite", "sf": 10, "tx_power_dbm": 14},
                            st["time_to_set_s"], "use_pass")

        if "load_rise" in kinds:
            cand = []
            for i in terr:
                dev = sim.devices[i]
                h = sim.ns.dev_history.get(i, [])
                if dev.base.sf == 7 and h and h[-1].snr_db - SNR_THRESHOLDS_DB[8] >= 5.0:
                    cand.append(i)
            self._issue(sim, cand[::2], {"sf": 8}, lease, "spread_sf")

    def extra_metrics(self) -> dict[str, Any]:
        return {"rule_invocations": self.n_invocations, "rule_actions": len(self.actions),
                "adr_changes": self.changes}
