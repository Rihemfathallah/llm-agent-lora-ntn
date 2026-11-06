"""Controller interface shared by every policy.

A controller is attached to one simulation run and reacts to three hooks:

* ``attach(sim)``              initial (base) parameters of every device
* ``on_uplink(sim, dev, rec)`` network side, after each *delivered* uplink;
                               may change the device's base parameters
                               (they take effect at once: class-A downlink)
* ``on_tick(sim, now, alerts)`` every detector tick (60 s); may issue
                               overrides through ``sim.issue_override``

Per-packet controllers (the oracle) set ``per_packet = True`` and implement
``choose(sim, dev, k)``, called at every reading.
"""

from __future__ import annotations

from typing import Any

from netsim.devices import TxParams

MIN_TX_POWER_DBM = 2
MAX_TX_POWER_DBM = 14


class Controller:
    name = "base"
    per_packet = False

    def attach(self, sim: Any) -> None:
        raise NotImplementedError

    def on_uplink(self, sim: Any, dev: Any, rec: Any) -> None:
        pass

    def on_tick(self, sim: Any, now: float, alerts: list) -> None:
        pass

    def choose(self, sim: Any, dev: Any, k: int) -> TxParams:
        raise NotImplementedError

    def extra_metrics(self) -> dict[str, Any]:
        return {}
