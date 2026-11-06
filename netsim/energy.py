"""Transmit energy model: E = V · I_tx(P_tx) · ToA.

Supply current from the Semtech SX1276 datasheet, Table 6 (power consumption,
typical values at VDD = 3.3 V, 25 °C), "IDDT, supply current in transmit mode
with impedance matching":

    RFOP = +7 dBm  on RFO_LF/HF pin   20 mA
    RFOP = +13 dBm on RFO_LF/HF pin   29 mA
    RFOP = +17 dBm on PA_BOOST        87 mA
    RFOP = +20 dBm on PA_BOOST       120 mA

EU868 limits the device to 14 dBm, which the RFO pin can deliver, so the
2–14 dBm range uses the two RFO points with linear interpolation, extended
linearly outside 7–13 dBm (assumption A9: the datasheet gives no value below
7 dBm or at 14 dBm on RFO).  The first version used a table (29 mA at 2 dBm
… 82 mA at 14 dBm) that does not match the datasheet.
"""

from __future__ import annotations

SUPPLY_VOLTAGE = 3.3          # V, datasheet typical conditions

_RFO_POINTS = ((7.0, 20.0), (13.0, 29.0))     # (dBm, mA) from the datasheet
_SLOPE = (_RFO_POINTS[1][1] - _RFO_POINTS[0][1]) / (_RFO_POINTS[1][0] - _RFO_POINTS[0][0])


def tx_current_ma(p_tx_dbm: float) -> float:
    """SX1276 supply current (mA) at output power *p_tx_dbm* on the RFO pin."""
    if not 2.0 <= p_tx_dbm <= 14.0:
        raise ValueError(f"EU868 device power must be 2–14 dBm, got {p_tx_dbm}")
    return _RFO_POINTS[0][1] + _SLOPE * (p_tx_dbm - _RFO_POINTS[0][0])


def tx_energy_joules(p_tx_dbm: float, toa_s: float) -> float:
    """Transmit energy (J) of one packet."""
    return SUPPLY_VOLTAGE * tx_current_ma(p_tx_dbm) * 1e-3 * toa_s
