"""Oracle (upper bound), section 3.5 phase 4.

At every reading the oracle looks at the *true* upcoming channel of that
packet (path loss, shadowing, this packet's fading, LOS/NLOS state,
jamming, deep fade, gateway state, satellite geometry and Doppler rate) and
tries every (link, channel, SF, power).  It picks the lowest-energy setting
with PER ≤ 1 %.  If none reaches 1 %, it picks the setting with the lowest
PER; if no gateway can receive at all, it does not transmit ("wait").

It ignores collisions (it does not know what the other devices will do),
and it is not subject to the class-A control constraint.
"""

from __future__ import annotations

import math
from typing import Any

from netsim.devices import TxParams
from netsim.energy import tx_energy_joules
from phy.ser_montecarlo import threshold_per
from policies.base import Controller


class OracleController(Controller):
    name = "Oracle"
    per_packet = True

    def __init__(self, target_per: float = 0.01) -> None:
        self.target = target_per

    def setup(self, sim: Any) -> None:
        self.powers = sorted(int(p) for p in sim.cfg["phy"]["tx_powers_dbm"])
        self.sfs = [int(s) for s in sim.cfg["phy"]["sfs"]]

    def attach(self, sim: Any) -> None:
        self.setup(sim)
        for dev in sim.devices:
            dev.base = TxParams(12, 14, (), dev.home_link)

    def _per(self, sim: Any, sf: int, snr: float, rate: float) -> float:
        return sim.per_lookup(sf, snr, rate)

    def best(self, sim: Any, dev: Any, k: int, t: float,
             links: tuple[str, ...] = ("terrestrial", "satellite"),
             channels: tuple[float, ...] | None = None) -> TxParams:
        """Best setting for reading *k* of *dev* at time *t*."""
        channels = channels or tuple(sim.channels)
        best_ok, best_e = None, math.inf
        best_any, best_any_key = None, (math.inf, math.inf)
        for link in links:
            if link == "satellite" and not sim.sat_enabled:
                continue
            for ch in channels if link == "terrestrial" else channels[:1]:
                for sf in self.sfs:
                    toa = sim.toa[sf]
                    # SNR at 0 dBm in the middle of the packet
                    snr0, _, _, rate = sim.true_snr(dev, k, link, 0.0, ch, t + 0.5 * toa)
                    if snr0 == -math.inf:
                        continue
                    if callable(getattr(sim.per_lookup, "min_snr", None)):
                        need = sim.per_lookup.min_snr(sf, rate, self.target)
                    else:
                        need = math.inf
                        for p in self.powers:
                            if threshold_per(sf, snr0 + p, rate) <= self.target:
                                need = snr0 + p
                                break
                    for p in self.powers:
                        snr = snr0 + p
                        if snr >= need:
                            e = tx_energy_joules(p, toa)
                            if e < best_e:
                                best_ok, best_e = TxParams(sf, p, (ch,), link), e
                            break
                    # track the most reliable option in case nothing reaches the target
                    p = self.powers[-1]
                    per = self._per(sim, sf, snr0 + p, rate)
                    key = (per, tx_energy_joules(p, toa))
                    if key < best_any_key:
                        best_any, best_any_key = TxParams(sf, p, (ch,), link), key
        if best_ok is not None:
            return best_ok
        if best_any is not None and best_any_key[0] < 1.0:
            return best_any
        return TxParams(12, 14, (), "wait")

    def choose(self, sim: Any, dev: Any, k: int) -> TxParams:
        # evaluate at the middle of the packet the device would send now
        return self.best(sim, dev, k, sim.now)

