"""Adaptive Data Rate (ADR) policy — Semtech's recommended algorithm.

Section 3.5 Phase 4, quoting the specification:

    After 20 uplinks:
        margin  = max(SNR of last 20) - SNR_req(SF) - 10 dB
        N_step  = floor(margin / 3)

        While N_step > 0 and SF > 7:
            lower SF by 1
            N_step -= 1

        While N_step > 0 and power above minimum:
            lower power by 3 dB
            N_step -= 1

        If N_step < 0:
            raise power toward maximum
            (one step of +3 dB per negative step, clamped at max)

EU868 constraints (Section 3.2):
    SF:    7 – 12
    Power: 2 – 14 dBm
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

from phy.css import SNR_THRESHOLDS_DB
from policies.fixed import BasePolicy, TxParams

# EU868 power limits
MIN_TX_POWER_DBM = 2
MAX_TX_POWER_DBM = 14

# ADR tunables
ADR_HISTORY_LEN = 20       # number of uplinks before first ADR decision
ADR_MARGIN_DB = 10.0       # installation margin
ADR_STEP_DB = 3            # power adjustment step (dB)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-device ADR state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _DeviceState:
    """Mutable state kept by ADR for each device."""
    sf: int = 12
    tx_power_dbm: int = MAX_TX_POWER_DBM
    snr_history: deque = None          # last ADR_HISTORY_LEN SNR values

    def __post_init__(self) -> None:
        if self.snr_history is None:
            self.snr_history = deque(maxlen=ADR_HISTORY_LEN)


# ═══════════════════════════════════════════════════════════════════════════
#  ADR Policy
# ═══════════════════════════════════════════════════════════════════════════

class ADRPolicy(BasePolicy):
    """Semtech ADR algorithm.

    The policy is conservative: devices start at SF12 / max power and
    converge downward as SNR history accumulates.  This mirrors real
    LoRaWAN network-server behavior.
    """

    name = "ADR"

    def __init__(
        self,
        initial_sf: int = 12,
        initial_power_dbm: int = MAX_TX_POWER_DBM,
        channel: float = 868.1,
        link: str = "terrestrial",
        history_len: int = ADR_HISTORY_LEN,
        margin_db: float = ADR_MARGIN_DB,
        step_db: int = ADR_STEP_DB,
    ) -> None:
        self.initial_sf = initial_sf
        self.initial_power = initial_power_dbm
        self.channel = channel
        self.link = link
        self.history_len = history_len
        self.margin_db = margin_db
        self.step_db = step_db

        self._devices: dict[int, _DeviceState] = {}

    # ── helpers ────────────────────────────────────────────────────────

    def _get_state(self, device_id: int) -> _DeviceState:
        if device_id not in self._devices:
            self._devices[device_id] = _DeviceState(
                sf=self.initial_sf,
                tx_power_dbm=self.initial_power,
                snr_history=deque(maxlen=self.history_len),
            )
        return self._devices[device_id]

    # ── ADR core ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_adr(
        snr_history: deque,
        current_sf: int,
        current_power: int,
        margin_db: float,
        step_db: int,
    ) -> tuple[int, int]:
        """Run one ADR iteration.

        Returns
        -------
        (new_sf, new_power_dbm)
        """
        if len(snr_history) < ADR_HISTORY_LEN:
            # Not enough history yet — keep current settings
            return current_sf, current_power

        # Step 1: compute margin
        snr_max = max(snr_history)
        snr_req = SNR_THRESHOLDS_DB[current_sf]
        margin = snr_max - snr_req - margin_db
        n_step = math.floor(margin / step_db)

        sf = current_sf
        power = current_power

        # Step 2: while N_step > 0 and SF > 7, lower SF
        while n_step > 0 and sf > 7:
            sf -= 1
            n_step -= 1

        # Step 3: while N_step > 0 and power above minimum, lower power
        while n_step > 0 and power > MIN_TX_POWER_DBM:
            power = max(power - step_db, MIN_TX_POWER_DBM)
            n_step -= 1

        # Step 4: if N_step < 0, raise power toward maximum
        if n_step < 0:
            steps_up = abs(n_step)
            power = min(power + steps_up * step_db, MAX_TX_POWER_DBM)

        return sf, power

    # ── policy interface ──────────────────────────────────────────────

    def decide(self, device_id: int, context: dict | None = None) -> TxParams:
        """Return current (SF, power) for the device.

        ADR adjustments happen in :meth:`update` after each uplink, not
        here.  ``decide`` simply returns the latest computed parameters.
        """
        state = self._get_state(device_id)
        return TxParams(
            sf=state.sf,
            tx_power_dbm=state.tx_power_dbm,
            channel=self.channel,
            link=self.link,
        )

    def update(self, device_id: int, snr_db: float, delivered: bool) -> None:
        """Feed back the SNR of the latest uplink and re-run ADR.

        Only *delivered* packets contribute to the SNR history (the
        gateway must have received the packet to measure its SNR).
        """
        state = self._get_state(device_id)
        if delivered:
            state.snr_history.append(snr_db)

        state.sf, state.tx_power_dbm = self._compute_adr(
            state.snr_history,
            state.sf,
            state.tx_power_dbm,
            self.margin_db,
            self.step_db,
        )

    def reset(self) -> None:
        """Clear all per-device state."""
        self._devices.clear()

    # ── introspection ─────────────────────────────────────────────────

    def get_device_state(self, device_id: int) -> dict:
        """Return a snapshot of the ADR state for a device (for logging)."""
        s = self._get_state(device_id)
        return {
            "sf": s.sf,
            "tx_power_dbm": s.tx_power_dbm,
            "history_len": len(s.snr_history),
            "snr_max": max(s.snr_history) if s.snr_history else None,
            "snr_min": min(s.snr_history) if s.snr_history else None,
        }
