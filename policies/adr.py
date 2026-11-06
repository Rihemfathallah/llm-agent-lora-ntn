"""Adaptive Data Rate (ADR), based on Semtech's recommended algorithm.

After 20 uplinks (section 3.5 phase 4):

    margin = max(SNR of last 20) − SNR_req(SF) − 10 dB
    N_step = floor(margin / 3)
    while N_step > 0 and SF > 7:          SF −= 1;  N_step −= 1
    while N_step > 0 and P > P_min:       P −= 3 dB; N_step −= 1
    while N_step < 0 and P < P_max:       P += 3 dB; N_step += 1

ADR never raises the SF: in LoRaWAN that only happens through the
device-side back-off after 64 + 32 uplinks without downlink (≈16 h at one
uplink per 10 min), which is outside the simulated horizon and not modelled.

The history is cleared when the parameters change (configurable), so every
decision is based on SNRs measured with the current power.

Warm start (``network.warm_start``): each terrestrial device starts from the
fixed point that ADR reaches on 20 past uplinks drawn from its own channel
(separate random stream), with a full history at the converged power.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from netsim.devices import TxParams, warm_snr_history
from phy.css import SNR_THRESHOLDS_DB
from policies.base import MAX_TX_POWER_DBM, MIN_TX_POWER_DBM, Controller


def adr_step(snr_history: list[float] | deque, sf: int, power: int,
             margin_db: float = 10.0, step_db: int = 3, history_len: int = 20) -> tuple[int, int]:
    """One iteration of the ADR rule; returns (new_sf, new_power)."""
    if len(snr_history) < history_len:
        return sf, power
    margin = max(snr_history) - SNR_THRESHOLDS_DB[sf] - margin_db
    n_step = math.floor(margin / step_db)
    while n_step > 0 and sf > 7:
        sf -= 1
        n_step -= 1
    while n_step > 0 and power > MIN_TX_POWER_DBM:
        power = max(power - step_db, MIN_TX_POWER_DBM)
        n_step -= 1
    while n_step < 0 and power < MAX_TX_POWER_DBM:
        power = min(power + step_db, MAX_TX_POWER_DBM)
        n_step += 1
    return sf, power


class ADRController(Controller):
    name = "ADR"

    def __init__(self) -> None:
        self.hist: dict[int, deque] = {}
        self.changes = 0

    def attach(self, sim: Any) -> None:
        a = sim.cfg["adr"]
        self.margin, self.step, self.n = float(a["margin_db"]), int(a["step_db"]), int(a["history_len"])
        self.reset_on_change = bool(a["reset_history_on_change"])
        warm = bool(sim.cfg["network"].get("warm_start", False))
        for dev in sim.devices:
            sf, pw = int(a["initial_sf"]), int(a["initial_power_dbm"])
            h: deque = deque(maxlen=self.n)
            if warm and dev.home_link == "terrestrial":
                base_hist = warm_snr_history(dev, sim.cfg, sim.seed, self.n, tx_power_dbm=0.0)
                for _ in range(10):                       # iterate to the fixed point
                    cur = [s + pw for s in base_hist]
                    nsf, npw = adr_step(cur, sf, pw, self.margin, self.step, self.n)
                    if (nsf, npw) == (sf, pw):
                        break
                    sf, pw = nsf, npw
                h.extend(s + pw for s in base_hist)
            self.hist[dev.device_id] = h
            dev.base = TxParams(sf, pw, (), dev.home_link)

    def on_uplink(self, sim: Any, dev: Any, rec: Any) -> None:
        h = self.hist[dev.device_id]
        # The network server knows the power it commanded.  If an override
        # (agent / rules) made the device transmit at another power, refer the
        # SNR back to ADR's own base power; otherwise ADR would lower its base
        # power after a boost and the device would come back under-powered.
        h.append(rec.snr_db - (rec.tx_power_dbm - dev.base.tx_power_dbm))
        sf, pw = adr_step(h, dev.base.sf, dev.base.tx_power_dbm, self.margin, self.step, self.n)
        if (sf, pw) != (dev.base.sf, dev.base.tx_power_dbm):
            sim.set_base(dev, dev.base.replace(sf=sf, tx_power_dbm=pw))
            self.changes += 1
            if self.reset_on_change:
                h.clear()

    def extra_metrics(self) -> dict[str, Any]:
        return {"adr_changes": self.changes}
