"""Unified link-budget module for terrestrial and satellite LoRa links.

Combines the Okumura–Hata terrestrial path-loss model (Section 3.4) with the
LEO satellite channel model (Section 3.5 Phase 2) into a single module that
exposes helpers for both link types and a generic ``link_budget`` function.

Assumptions (to be stated in the report):
  - Okumura–Hata model is used for 868 MHz (outside the strict 150–1500 MHz
    validity range, but within common engineering practice for sub-GHz ISM).
  - Rural (open area) correction is applied to the urban Hata loss.
  - Log-normal shadow fading is i.i.d. per link realisation (terrestrial) or
    per time step (satellite pass).
  - Antenna gains default to 0 dBi for terrestrial (omnidirectional devices).
"""

import math
from typing import Any

import numpy as np

from channel.satellite import (
    slant_range,
    fspl as _sat_fspl,
    clutter_loss,
    shadow_fading_sample,
    satellite_link_budget,
    simulate_pass,           # re-exported for convenience
    DEFAULT_ALTITUDE,
    DEFAULT_FC,
)
from phy.css import SNR_THRESHOLDS_DB


# ---------------------------------------------------------------------------
# Default scenario parameters
# ---------------------------------------------------------------------------

_DEFAULT_F_MHZ = 868.0          # carrier frequency (MHz)
_DEFAULT_H_B = 30.0             # base-station / gateway height (m)
_DEFAULT_H_M = 1.5              # mobile / device height (m)
_DEFAULT_BW = 125_000.0         # bandwidth (Hz)
_DEFAULT_NF = 6.0               # receiver noise figure (dB)
_DEFAULT_SHADOW_STD = 7.0       # terrestrial shadow-fading std dev (dB)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Okumura–Hata terrestrial path-loss model
# ═══════════════════════════════════════════════════════════════════════════

def hata_correction_factor(f_mhz: float, h_m: float) -> float:
    """Mobile-station antenna height correction factor a(h_m).

    Small/medium city formula (valid for all environments in the rural
    correction path):

        a(h_m) = (1.1·log10(f) − 0.7)·h_m − (1.56·log10(f) − 0.8)

    Parameters
    ----------
    f_mhz : float
        Carrier frequency in MHz  (150–1500 MHz nominal).
    h_m : float
        Mobile-station antenna height in metres (1–10 m).

    Returns
    -------
    float
        Correction factor a(h_m) in dB.
    """
    log_f = math.log10(f_mhz)
    return (1.1 * log_f - 0.7) * h_m - (1.56 * log_f - 0.8)


def hata_urban_loss(
    f_mhz: float,
    h_b: float,
    h_m: float,
    d_km: float,
) -> float:
    """Okumura–Hata median path loss for an urban environment (dB).

        L_urban = 69.55 + 26.16·log10(f) − 13.82·log10(h_b)
                  − a(h_m) + (44.9 − 6.55·log10(h_b))·log10(d)

    Valid ranges: 150–1500 MHz, h_b 30–200 m, h_m 1–10 m, d 1–20 km.

    Parameters
    ----------
    f_mhz : float
        Carrier frequency in MHz.
    h_b : float
        Base-station antenna height in metres.
    h_m : float
        Mobile-station antenna height in metres.
    d_km : float
        Distance between base and mobile in km.

    Returns
    -------
    float
        Urban median path loss in dB.
    """
    log_f = math.log10(f_mhz)
    log_hb = math.log10(h_b)
    log_d = math.log10(d_km)
    a_hm = hata_correction_factor(f_mhz, h_m)
    return (
        69.55
        + 26.16 * log_f
        - 13.82 * log_hb
        - a_hm
        + (44.9 - 6.55 * log_hb) * log_d
    )


def hata_rural_loss(
    f_mhz: float,
    h_b: float,
    h_m: float,
    d_km: float,
) -> float:
    """Okumura–Hata path loss for a rural (open area) environment (dB).

        L_rural = L_urban − 4.78·(log10(f))² + 18.33·log10(f) − 40.94

    Parameters
    ----------
    f_mhz : float
        Carrier frequency in MHz.
    h_b : float
        Base-station antenna height in metres.
    h_m : float
        Mobile-station antenna height in metres.
    d_km : float
        Distance in km.

    Returns
    -------
    float
        Rural median path loss in dB.
    """
    l_urban = hata_urban_loss(f_mhz, h_b, h_m, d_km)
    log_f = math.log10(f_mhz)
    return l_urban - 4.78 * log_f ** 2 + 18.33 * log_f - 40.94


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Generic link budget
# ═══════════════════════════════════════════════════════════════════════════

def link_budget(
    p_tx_dbm: float,
    g_tx_dbi: float,
    g_rx_dbi: float,
    l_path_db: float,
    l_shadow_db: float = 0.0,
    l_clutter_db: float = 0.0,
    bw: float = _DEFAULT_BW,
    nf_db: float = _DEFAULT_NF,
) -> float:
    """Unified link-budget formula – returns received SNR in dB.

        SNR = P_tx + G_tx + G_rx − L_path − L_shadow − L_clutter
              − (−174 + 10·log10(BW) + NF)

    Parameters
    ----------
    p_tx_dbm : float
        Transmit power (dBm).
    g_tx_dbi : float
        Transmit antenna gain (dBi).
    g_rx_dbi : float
        Receive antenna gain (dBi).
    l_path_db : float
        Total path loss (dB).
    l_shadow_db : float
        Shadow-fading loss (dB, positive = loss).
    l_clutter_db : float
        Clutter/environment loss (dB).
    bw : float
        Bandwidth in Hz.
    nf_db : float
        Receiver noise figure (dB).

    Returns
    -------
    float
        SNR in dB.
    """
    noise_floor = -174.0 + 10.0 * math.log10(bw) + nf_db
    return (
        p_tx_dbm + g_tx_dbi + g_rx_dbi
        - l_path_db - l_shadow_db - l_clutter_db
        - noise_floor
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Terrestrial SNR
# ═══════════════════════════════════════════════════════════════════════════

def terrestrial_snr(
    p_tx_dbm: float,
    f_mhz: float = _DEFAULT_F_MHZ,
    d_km: float = 5.0,
    h_b: float = _DEFAULT_H_B,
    h_m: float = _DEFAULT_H_M,
    bw: float = _DEFAULT_BW,
    nf_db: float = _DEFAULT_NF,
    shadow_std_db: float = _DEFAULT_SHADOW_STD,
    rng: np.random.Generator | None = None,
) -> float:
    """Compute terrestrial link SNR (dB) using rural Okumura–Hata.

    Steps:
      1. Deterministic path loss via ``hata_rural_loss``.
      2. Log-normal shadow fading drawn from N(0, shadow_std_db).
      3. SNR via the generic ``link_budget`` with 0 dBi antennas.

    Parameters
    ----------
    p_tx_dbm : float
        Transmit power (dBm).
    f_mhz : float
        Carrier frequency in MHz.
    d_km : float
        Distance in km.
    h_b : float
        Base-station antenna height (m).
    h_m : float
        Device antenna height (m).
    bw : float
        Bandwidth in Hz.
    nf_db : float
        Receiver noise figure (dB).
    shadow_std_db : float
        Standard deviation of log-normal shadow fading (dB).
    rng : numpy.random.Generator or None
        Random number generator for shadow fading.

    Returns
    -------
    float
        Received SNR in dB.
    """
    if rng is None:
        rng = np.random.default_rng()

    l_path = hata_rural_loss(f_mhz, h_b, h_m, d_km)
    l_shadow = rng.normal(0.0, shadow_std_db)

    return link_budget(
        p_tx_dbm=p_tx_dbm,
        g_tx_dbi=0.0,
        g_rx_dbi=0.0,
        l_path_db=l_path,
        l_shadow_db=l_shadow,
        l_clutter_db=0.0,
        bw=bw,
        nf_db=nf_db,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Satellite SNR (single snapshot)
# ═══════════════════════════════════════════════════════════════════════════

def satellite_snr(
    p_tx_dbm: float,
    g_tx_dbi: float = 0.0,
    g_rx_dbi: float = 6.0,
    elevation_deg: float = 90.0,
    h_km: float = DEFAULT_ALTITUDE,
    f_c: float = DEFAULT_FC,
    bw: float = _DEFAULT_BW,
    nf_db: float = _DEFAULT_NF,
    rng: np.random.Generator | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute satellite link SNR at a single elevation angle.

    Combines free-space path loss, 3GPP clutter loss, and elevation-dependent
    shadow fading using the helpers in ``channel.satellite``.

    Parameters
    ----------
    p_tx_dbm : float
        Transmit power (dBm).
    g_tx_dbi : float
        Ground-station antenna gain (dBi).
    g_rx_dbi : float
        Satellite receive antenna gain (dBi).
    elevation_deg : float
        Elevation angle (degrees).
    h_km : float
        Satellite altitude (km).
    f_c : float
        Carrier frequency (Hz).
    bw : float
        Bandwidth (Hz).
    nf_db : float
        Receiver noise figure (dB).
    rng : numpy.random.Generator or None
        RNG for shadow-fading sample.

    Returns
    -------
    (snr_db, breakdown) : tuple[float, dict]
        ``snr_db`` – received SNR in dB.
        ``breakdown`` – dict with keys ``range_km``, ``fspl_db``,
        ``clutter_db``, ``shadow_db``, ``noise_floor_db``.
    """
    if rng is None:
        rng = np.random.default_rng()

    f_mhz = f_c / 1e6
    d_km = slant_range(elevation_deg, h_km)
    l_fspl = _sat_fspl(d_km, f_mhz)
    l_clutter = float(clutter_loss(elevation_deg))
    l_shadow = float(shadow_fading_sample(elevation_deg, rng))
    noise_floor = -174.0 + 10.0 * math.log10(bw) + nf_db

    snr_db = satellite_link_budget(
        p_tx_dbm=p_tx_dbm,
        g_tx_dbi=g_tx_dbi,
        g_rx_dbi=g_rx_dbi,
        fspl_db=l_fspl,
        shadow_db=l_shadow,
        clutter_db=l_clutter,
        bw=bw,
        nf_db=nf_db,
    )

    breakdown: dict[str, float] = {
        "range_km": d_km,
        "fspl_db": l_fspl,
        "clutter_db": l_clutter,
        "shadow_db": l_shadow,
        "noise_floor_db": noise_floor,
    }
    return snr_db, breakdown


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Satellite pass SNR (time-series)
# ═══════════════════════════════════════════════════════════════════════════

def satellite_pass_snr(
    pass_data: dict[str, np.ndarray],
    p_tx_dbm: float = 14.0,
    g_tx_dbi: float = 0.0,
    g_rx_dbi: float = 6.0,
    bw: float = _DEFAULT_BW,
    nf_db: float = _DEFAULT_NF,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute SNR over an entire satellite pass.

    Takes the output dict of ``channel.satellite.simulate_pass`` and computes
    the received SNR at each time step, including per-step shadow-fading
    realisations.

    Parameters
    ----------
    pass_data : dict
        Output of ``simulate_pass``, containing at least ``t_s``,
        ``elevation_deg``, ``fspl_db``, ``doppler_hz``,
        ``doppler_rate_hz_s``.
    p_tx_dbm : float
        Transmit power (dBm).
    g_tx_dbi : float
        Ground-station antenna gain (dBi).
    g_rx_dbi : float
        Satellite receive antenna gain (dBi).
    bw : float
        Bandwidth (Hz).
    nf_db : float
        Receiver noise figure (dB).
    seed : int or None
        Seed for the shadow-fading RNG (reproducibility).

    Returns
    -------
    dict
        ``t_s``               – time array (s), same as input.
        ``elevation_deg``     – elevation array (deg).
        ``snr_db``            – received SNR (dB) at each step.
        ``doppler_hz``        – Doppler shift (Hz).
        ``doppler_rate_hz_s`` – Doppler rate (Hz/s).
    """
    rng = np.random.default_rng(seed)
    elevation = pass_data["elevation_deg"]
    n_steps = len(elevation)

    noise_floor = -174.0 + 10.0 * math.log10(bw) + nf_db

    # Vectorised clutter loss
    l_clutter = clutter_loss(elevation)                   # ndarray

    # Per-step shadow-fading samples
    l_shadow = np.array(
        [float(shadow_fading_sample(float(el), rng)) for el in elevation],
        dtype=np.float64,
    )

    # SNR at each step
    snr_db = (
        p_tx_dbm
        + g_tx_dbi
        + g_rx_dbi
        - pass_data["fspl_db"]
        - l_shadow
        - l_clutter
        - noise_floor
    )

    return {
        "t_s": pass_data["t_s"],
        "elevation_deg": elevation,
        "snr_db": snr_db,
        "doppler_hz": pass_data["doppler_hz"],
        "doppler_rate_hz_s": pass_data["doppler_rate_hz_s"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Feasible spreading factors
# ═══════════════════════════════════════════════════════════════════════════

def feasible_sfs(
    snr_db: float,
    doppler_rate_hz_s: float = 0.0,
    per_table: dict[int, Any] | None = None,
) -> list[int]:
    """Return the list of spreading factors that can close the link.

    For each SF (7–12), the link is feasible if:
      - ``snr_db >= SNR_THRESHOLDS_DB[sf]``  (simple threshold check), or
      - ``per_table[sf]`` maps to a PER < 10 % at the given SNR/Doppler rate
        (when ``per_table`` is provided).

    Parameters
    ----------
    snr_db : float
        Received SNR (dB).
    doppler_rate_hz_s : float
        Doppler rate (Hz/s), reserved for PER-table look-ups.
    per_table : dict or None
        Optional mapping ``{sf: callable(snr_db, doppler_rate_hz_s) -> PER}``.
        When ``None``, the simple threshold check is used.

    Returns
    -------
    list[int]
        Sorted list of feasible SFs.
    """
    feasible: list[int] = []
    for sf in sorted(SNR_THRESHOLDS_DB):
        if per_table is not None:
            per = per_table[sf](snr_db, doppler_rate_hz_s)
            if per < 0.10:
                feasible.append(sf)
        else:
            if snr_db >= SNR_THRESHOLDS_DB[sf]:
                feasible.append(sf)
    return feasible
