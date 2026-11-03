"""Satellite channel: LEO orbit geometry, pass model, Doppler and link budget.

All formulas follow Section 3.4 of the project specification.

Assumptions (to be stated in the report):
  - Earth's rotation is ignored.
  - 2-D pass model: the satellite moves on a circle of radius R_E + h.
  - 3GPP TR 38.811 S-band rural values used as approximation for 868 MHz.
  - Atmospheric absorption neglected for sub-GHz at elevations > 10°.
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

R_E = 6_371.0               # Earth radius (km)
G_ME = 3.986004418e14       # G · M_E  (m³/s²)
C = 299_792_458.0            # speed of light (m/s)

# ---------------------------------------------------------------------------
# Default scenario parameters
# ---------------------------------------------------------------------------

DEFAULT_ALTITUDE = 600.0     # km
DEFAULT_FC = 868e6           # carrier frequency (Hz)
DEFAULT_MIN_ELEVATION = 10.0 # degrees


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Static geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def orbital_speed(h_km: float = DEFAULT_ALTITUDE) -> float:
    """Orbital speed (m/s).

        v = sqrt( G·M_E / (R_E + h) )

    ≈ 7.56 km/s at 600 km altitude.
    """
    r = (R_E + h_km) * 1e3          # orbit radius in metres
    return math.sqrt(G_ME / r)


def slant_range(elevation_deg: float, h_km: float = DEFAULT_ALTITUDE) -> float:
    """Slant range from ground station to satellite (km).

        d(ε) = sqrt( (R_E·sin ε)² + h² + 2·h·R_E ) − R_E·sin ε

    Parameters
    ----------
    elevation_deg : float
        Elevation angle ε in degrees.
    h_km : float
        Satellite altitude in km.

    Returns
    -------
    float
        Slant range in km.
    """
    eps = math.radians(elevation_deg)
    sin_e = math.sin(eps)
    return (
        math.sqrt((R_E * sin_e) ** 2 + h_km ** 2 + 2 * h_km * R_E)
        - R_E * sin_e
    )


def fspl(d_km: float, f_mhz: float) -> float:
    """Free-space path loss (dB).

        FSPL = 20·log10(d_km) + 20·log10(f_MHz) + 32.45
    """
    return 20.0 * math.log10(d_km) + 20.0 * math.log10(f_mhz) + 32.45


def propagation_delay(d_km: float) -> float:
    """One-way propagation delay (seconds).

        τ = d / c
    """
    return d_km * 1e3 / C


# ═══════════════════════════════════════════════════════════════════════════
# 2.  LEO pass model (2-D)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_pass(
    h_km: float = DEFAULT_ALTITUDE,
    f_c: float = DEFAULT_FC,
    min_elevation_deg: float = DEFAULT_MIN_ELEVATION,
    max_elevation_deg: float = 90.0,
    dt: float = 1.0,
) -> dict[str, np.ndarray]:
    """Simulate a full LEO overhead pass in 2-D.

    The satellite moves on a circle of radius R_E + h at angular speed
    ω = v / (R_E + h).  The ground station sits at the surface directly
    below the point of closest approach (zenith pass when
    max_elevation_deg = 90°).

    Earth's rotation is ignored (stated assumption).

    Returns arrays sampled every *dt* seconds while the satellite is above
    *min_elevation_deg*.

    Parameters
    ----------
    h_km : float
        Altitude in km.
    f_c : float
        Carrier frequency in Hz.
    min_elevation_deg : float
        Minimum usable elevation angle (degrees).
    max_elevation_deg : float
        Maximum elevation during the pass (degrees).  90° = zenith pass.
    dt : float
        Time step in seconds.

    Returns
    -------
    dict with keys
        ``t_s``            – time array centred on closest approach (s)
        ``elevation_deg``  – elevation angle (degrees)
        ``range_km``       – slant range (km)
        ``delay_s``        – propagation delay (s)
        ``doppler_hz``     – Doppler shift (Hz)
        ``doppler_rate_hz_s`` – Doppler rate (Hz/s)
        ``fspl_db``        – free-space path loss (dB)
    """
    r_orbit = R_E + h_km                          # km
    v = orbital_speed(h_km)                        # m/s
    omega = v / (r_orbit * 1e3)                    # rad/s

    # --- Determine the angular offset of the ground station ----------------
    # For a non-zenith pass (max_elevation < 90°) the ground station is
    # offset from the sub-satellite track.  We model this by placing the
    # station on the Earth's surface at angle α from the sub-satellite
    # point at closest approach, where α satisfies:
    #   tan(π/2 − max_elev) = (r_orbit·sin α − R_E·sin α) / (r_orbit·cos α − R_E)
    # For a zenith pass (max_elev = 90°), α = 0 and the geometry simplifies.
    #
    # For simplicity we keep the zenith-pass model (α = 0) and clip the pass
    # at the requested min_elevation.  max_elevation_deg is included for
    # future extension; currently only 90° is fully accurate.

    # --- Half-duration of the visible pass ---------------------------------
    # At the visibility limit the elevation angle equals min_elevation_deg.
    # We solve for the central angle θ_max at which elevation = min_elev.
    #
    # Geometry (2-D, origin = Earth centre, station at angle 0):
    #   satellite position:  (r_orbit·cos θ,  r_orbit·sin θ)
    #   station position:    (R_E, 0)
    #   elevation ε = arctan2(r_orbit·cos θ − R_E,  r_orbit·|sin θ|) − 0
    #   but more robustly via the law of sines on the triangle
    #   (Earth centre, station, satellite):
    #      sin(π/2 + ε) / r_orbit = sin(central angle) / d
    #
    # We use a direct geometric computation in the loop instead to avoid
    # edge cases.

    # Over-estimate the half-duration and then trim
    half_t_max = 15.0 * 60.0                       # generous 15-min half window
    n_half = int(math.ceil(half_t_max / dt))

    # Build the full time grid centred on t = 0 (closest approach)
    t = np.arange(-n_half, n_half + 1, dtype=np.float64) * dt

    # Satellite angular position (θ = 0 at zenith of the ground station)
    theta = omega * t                               # radians

    # Satellite Cartesian position (origin = Earth centre)
    sat_x = r_orbit * np.cos(theta)                 # km
    sat_y = r_orbit * np.sin(theta)                 # km

    # Ground station at (R_E, 0)
    gs_x, gs_y = R_E, 0.0

    # Slant-range vector
    dx = sat_x - gs_x
    dy = sat_y - gs_y
    d = np.sqrt(dx ** 2 + dy ** 2)                  # km

    # Elevation angle
    # The "up" direction at the station is the radial: (1, 0).
    # The look vector is (dx, dy); its angle with the local horizontal
    # (perpendicular to radial, i.e. tangent) gives the elevation.
    # elevation = arcsin( dot(look, radial_unit) / |look| )
    #           = arcsin( dx / d )  — but we must subtract π/2 offset
    # More carefully:
    #   angle between look vector and radial = arccos( dx / d )
    #   elevation = π/2 − that angle            (positive when sat above horizon)
    cos_zenith = dx / d
    elevation_rad = np.pi / 2.0 - np.arccos(np.clip(cos_zenith, -1.0, 1.0))
    elevation_deg = np.degrees(elevation_rad)

    # Mask to the visible portion of the pass
    mask = elevation_deg >= min_elevation_deg
    if not np.any(mask):
        # Satellite never rises above minimum elevation
        empty = np.array([], dtype=np.float64)
        return {
            "t_s": empty,
            "elevation_deg": empty,
            "range_km": empty,
            "delay_s": empty,
            "doppler_hz": empty,
            "doppler_rate_hz_s": empty,
            "fspl_db": empty,
        }

    t = t[mask]
    elevation_deg = elevation_deg[mask]
    d = d[mask]

    # Propagation delay
    delay = d * 1e3 / C                             # seconds

    # FSPL
    f_mhz = f_c / 1e6
    fspl_db = 20.0 * np.log10(d) + 20.0 * np.log10(f_mhz) + 32.45

    # --- Doppler shift  f_D(t) = −(f_c / c) · dd/dt -----------------------
    # Compute dd/dt numerically (central differences, forward/backward at edges)
    dd_dt_km_s = np.gradient(d, dt)                 # km/s
    dd_dt_m_s = dd_dt_km_s * 1e3                    # m/s
    doppler_hz = -(f_c / C) * dd_dt_m_s

    # --- Doppler rate  f_D'(t) = d(f_D)/dt  --------------------------------
    doppler_rate = np.gradient(doppler_hz, dt)      # Hz/s

    return {
        "t_s": t,
        "elevation_deg": elevation_deg,
        "range_km": d,
        "delay_s": delay,
        "doppler_hz": doppler_hz,
        "doppler_rate_hz_s": doppler_rate,
        "fspl_db": fspl_db,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  3GPP TR 38.811 extra satellite losses (S-band rural approximation)
# ═══════════════════════════════════════════════════════════════════════════

# Clutter loss (dB) by elevation angle — S-band, rural environment
# Interpolated from TR 38.811 Table 6.6.2-1 (representative values).
_CLUTTER_ELEV = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float64)
_CLUTTER_LOSS = np.array([3.8, 2.5, 1.7, 1.2, 0.8, 0.5, 0.3, 0.1, 0.0],
                         dtype=np.float64)

# Shadow fading std dev (dB) by elevation — S-band, rural, LOS dominant
_SHADOW_STD   = np.array([3.5, 3.0, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8],
                         dtype=np.float64)


def clutter_loss(elevation_deg: float | np.ndarray) -> float | np.ndarray:
    """Clutter loss (dB) as a function of elevation angle.

    Linearly interpolated from 3GPP TR 38.811 S-band rural values
    (used as approximation for 868 MHz).
    """
    return np.interp(elevation_deg, _CLUTTER_ELEV, _CLUTTER_LOSS)


def shadow_fading_std(elevation_deg: float | np.ndarray) -> float | np.ndarray:
    """Shadow-fading standard deviation (dB) as a function of elevation.

    Linearly interpolated from 3GPP TR 38.811 S-band rural values.
    """
    return np.interp(elevation_deg, _CLUTTER_ELEV, _SHADOW_STD)


def shadow_fading_sample(
    elevation_deg: float | np.ndarray,
    rng: np.random.Generator | None = None,
) -> float | np.ndarray:
    """Draw a log-normal shadow-fading sample (dB) for the satellite link.

    Returns a zero-mean Gaussian variate with elevation-dependent std dev.
    """
    if rng is None:
        rng = np.random.default_rng()
    std = shadow_fading_std(elevation_deg)
    return rng.normal(0.0, std)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Satellite link budget
# ═══════════════════════════════════════════════════════════════════════════

def satellite_link_budget(
    p_tx_dbm: float,
    g_tx_dbi: float,
    g_rx_dbi: float,
    fspl_db: float,
    shadow_db: float,
    clutter_db: float,
    bw: float = 125_000.0,
    nf_db: float = 6.0,
) -> float:
    """Compute received SNR (dB) for the satellite link.

        SNR = P_tx + G_tx + G_rx − L_path − L_shadow − L_clutter
              − (−174 + 10·log10(BW) + NF)

    Parameters
    ----------
    p_tx_dbm : float
        Transmit power (dBm).
    g_tx_dbi : float
        Transmit antenna gain (dBi).
    g_rx_dbi : float
        Satellite receive antenna gain (dBi).  Config parameter.
    fspl_db : float
        Free-space path loss (dB).
    shadow_db : float
        Shadow-fading loss (dB), positive = loss.
    clutter_db : float
        Clutter loss (dB).
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
    return p_tx_dbm + g_tx_dbi + g_rx_dbi - fspl_db - shadow_db - clutter_db - noise_floor
