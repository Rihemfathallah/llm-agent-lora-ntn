"""Fixed transmission policies (baselines).

Section 3.5 Phase 4:
  - Fixed SF12 at maximum power  (most robust on the ground, highest energy)
  - Fixed SF7                    (fastest, least robust)

All policies expose the same interface so they can be swapped in the
simulator without changing calling code.
"""

from __future__ import annotations

from dataclasses import dataclass

from phy.css import SNR_THRESHOLDS_DB

# EU868 constraints (Section 3.2)
MIN_TX_POWER_DBM = 2
MAX_TX_POWER_DBM = 14
DEFAULT_CHANNEL = 868.1       # MHz


# ═══════════════════════════════════════════════════════════════════════════
#  Common policy interface
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TxParams:
    """Transmission parameters chosen by a policy."""
    sf: int
    tx_power_dbm: int
    channel: float          # MHz
    link: str               # "terrestrial" | "satellite" | "wait"


class BasePolicy:
    """Abstract base for all transmission policies.

    Subclasses must implement :meth:`decide`.
    """

    name: str = "base"

    def decide(self, device_id: int, context: dict | None = None) -> TxParams:
        """Choose transmission parameters for *device_id*.

        Parameters
        ----------
        device_id : int
            The device requesting parameters.
        context : dict, optional
            Situational information (SNR history, satellite status,
            anomaly flags, …).  Its contents depend on the scenario.

        Returns
        -------
        TxParams
        """
        raise NotImplementedError

    def update(self, device_id: int, snr_db: float, delivered: bool) -> None:
        """Feedback after a transmission attempt (optional hook)."""

    def reset(self) -> None:
        """Reset internal state (e.g. between simulation runs)."""


# ═══════════════════════════════════════════════════════════════════════════
#  Fixed SF12 at maximum power
# ═══════════════════════════════════════════════════════════════════════════

class FixedSF12MaxPower(BasePolicy):
    """Always transmit at SF12 with maximum power (14 dBm).

    Most robust on the ground — highest energy cost.
    """

    name = "Fixed-SF12-MaxPwr"

    def __init__(self, channel: float = DEFAULT_CHANNEL,
                 link: str = "terrestrial") -> None:
        self.channel = channel
        self.link = link

    def decide(self, device_id: int, context: dict | None = None) -> TxParams:
        return TxParams(
            sf=12,
            tx_power_dbm=MAX_TX_POWER_DBM,
            channel=self.channel,
            link=self.link,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Fixed SF7 at maximum power
# ═══════════════════════════════════════════════════════════════════════════

class FixedSF7(BasePolicy):
    """Always transmit at SF7 with maximum power (14 dBm).

    Fastest airtime, lowest energy, least robust.
    """

    name = "Fixed-SF7"

    def __init__(self, channel: float = DEFAULT_CHANNEL,
                 link: str = "terrestrial",
                 tx_power_dbm: int = MAX_TX_POWER_DBM) -> None:
        self.channel = channel
        self.link = link
        self.tx_power_dbm = tx_power_dbm

    def decide(self, device_id: int, context: dict | None = None) -> TxParams:
        return TxParams(
            sf=7,
            tx_power_dbm=self.tx_power_dbm,
            channel=self.channel,
            link=self.link,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Fixed arbitrary SF / power (utility)
# ═══════════════════════════════════════════════════════════════════════════

class FixedPolicy(BasePolicy):
    """Transmit with a fixed (SF, power) pair — general-purpose baseline."""

    def __init__(
        self,
        sf: int = 7,
        tx_power_dbm: int = MAX_TX_POWER_DBM,
        channel: float = DEFAULT_CHANNEL,
        link: str = "terrestrial",
    ) -> None:
        if not 7 <= sf <= 12:
            raise ValueError(f"SF must be 7-12, got {sf}")
        if not MIN_TX_POWER_DBM <= tx_power_dbm <= MAX_TX_POWER_DBM:
            raise ValueError(f"Power must be {MIN_TX_POWER_DBM}-"
                             f"{MAX_TX_POWER_DBM} dBm, got {tx_power_dbm}")
        self.sf = sf
        self.tx_power_dbm = tx_power_dbm
        self.channel = channel
        self.link = link
        self.name = f"Fixed-SF{sf}-{tx_power_dbm}dBm"

    def decide(self, device_id: int, context: dict | None = None) -> TxParams:
        return TxParams(
            sf=self.sf,
            tx_power_dbm=self.tx_power_dbm,
            channel=self.channel,
            link=self.link,
        )
