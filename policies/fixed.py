"""Fixed transmission policies (baselines, section 3.5 phase 4).

* Fixed SF12 at maximum power: most robust on the ground, highest energy.
* Fixed SF7 at maximum power: fastest, least robust.

Devices use their deployment link (terrestrial, or satellite for devices
outside terrestrial coverage) and hop over all channels.
"""

from __future__ import annotations

from typing import Any

from netsim.devices import TxParams
from policies.base import MAX_TX_POWER_DBM, MIN_TX_POWER_DBM, Controller


class FixedController(Controller):
    def __init__(self, sf: int, tx_power_dbm: int = MAX_TX_POWER_DBM, name: str | None = None) -> None:
        if not 7 <= sf <= 12:
            raise ValueError(f"SF must be 7-12, got {sf}")
        if not MIN_TX_POWER_DBM <= tx_power_dbm <= MAX_TX_POWER_DBM:
            raise ValueError(f"power must be {MIN_TX_POWER_DBM}-{MAX_TX_POWER_DBM} dBm")
        self.sf = sf
        self.tx_power_dbm = tx_power_dbm
        self.name = name or f"Fixed-SF{sf}"

    def attach(self, sim: Any) -> None:
        for dev in sim.devices:
            dev.base = TxParams(self.sf, self.tx_power_dbm, (), dev.home_link)


def FixedSF12MaxPower() -> FixedController:
    return FixedController(12, MAX_TX_POWER_DBM, "Fixed-SF12")


def FixedSF7() -> FixedController:
    return FixedController(7, MAX_TX_POWER_DBM, "Fixed-SF7")
