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
# 3.  3GPP TR 38.811 extra satellite losses (S-band, suburban & rural)
# ═══════════════════════════════════════════════════════════════════════════
#
# Values copied from 3GPP TR 38.811 V15.1.0:
#   Table 6.6.1-1  LOS probability, "Suburban and Rural scenarios"
#   Table 6.6.2-3  Shadow fading and clutter loss, suburban and rural, S-band
# The report gives S-band and Ka-band only; S-band is used as an
# approximation for 868 MHz (assumption A2).  The first version of this file
# used invented "representative" values (clutter 3.8 dB at 10°, 0 at 90°).
# In TR 38.811 clutter loss applies to NLOS only and is 16–19.5 dB.

_TR_ELEV = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float64)
_TR_P_LOS = np.array([0.782, 0.869, 0.919, 0.929, 0.935, 0.940, 0.949, 0.952, 0.998])
_TR_SF_LOS = np.array([1.79, 1.14, 1.14, 0.92, 1.42, 1.56, 0.85, 0.72, 0.72])
_TR_SF_NLOS = np.array([8.93, 9.08, 8.78, 10.25, 10.56, 10.74, 10.17, 11.52, 11.52])
_TR_CL_NLOS = np.array([19.52, 18.17, 18.42, 18.28, 18.63, 17.68, 16.50, 16.30, 16.30])


def los_probability(elevation_deg: float | np.ndarray) -> float | np.ndarray:
    """LOS probability (TR 38.811 Table 6.6.1-1, suburban/rural), interpolated."""
    return np.interp(elevation_deg, _TR_ELEV, _TR_P_LOS)


def clutter_loss(elevation_deg: float | np.ndarray, los: bool | np.ndarray = False):
    """Clutter loss (dB): 0 in LOS, TR 38.811 S-band rural CL in NLOS."""
    cl = np.interp(elevation_deg, _TR_ELEV, _TR_CL_NLOS)
    return np.where(los, 0.0, cl)


def shadow_fading_std(elevation_deg: float | np.ndarray, los: bool | np.ndarray = True):
    """Shadow-fading standard deviation (dB) for LOS or NLOS."""
    return np.where(
        los,
        np.interp(elevation_deg, _TR_ELEV, _TR_SF_LOS),
        np.interp(elevation_deg, _TR_ELEV, _TR_SF_NLOS),
    )


def expected_extra_loss(elevation_deg: float | np.ndarray):
    """Mean extra loss (dB) = P(NLOS) · CL_NLOS, for link-budget figures."""
    p = los_probability(elevation_deg)
    return (1.0 - p) * np.interp(elevation_deg, _TR_ELEV, _TR_CL_NLOS)


def extra_loss_sample(
    elevation_deg: float,
    u_los: float,
    z: float,
) -> tuple[float, bool]:
    """Draw the TR 38.811 extra loss (dB) from given random numbers.

    *u_los* ~ U(0,1) decides LOS (u < P_LOS); *z* ~ N(0,1) scales the
    shadow fading.  Passing the random numbers in (instead of an RNG) lets
    every policy see the same channel realisation.
    Returns (loss_db, is_los).
    """
    los = bool(u_los < float(los_probability(elevation_deg)))
    std = float(shadow_fading_std(elevation_deg, los))
    return float(clutter_loss(elevation_deg, los)) + std * z, los


def shadow_fading_sample(
    elevation_deg: float | np.ndarray,
    rng: np.random.Generator | None = None,
) -> float | np.ndarray:
    """Draw a LOS/NLOS mixture extra loss (dB) for the satellite link."""
    if rng is None:
        rng = np.random.default_rng()
    loss, _ = extra_loss_sample(float(elevation_deg), rng.random(), rng.standard_normal())
    return loss


# ═══════════════════════════════════════════════════════════════════════════
# 3b. Pass geometry for any maximum elevation (analytic, no Earth rotation)
# ═══════════════════════════════════════════════════════════════════════════

def central_angle_for_elevation(elevation_deg: float, h_km: float = DEFAULT_ALTITUDE) -> float:
    """Earth central angle (rad) between ground point and sub-satellite point."""
    eps = math.radians(elevation_deg)
    return math.acos(R_E * math.cos(eps) / (R_E + h_km)) - eps


class SatellitePass:
    """One LEO pass over a ground point, closed-form geometry.

    The ground point is at angular distance β from the orbital plane, with
    β chosen so that the elevation at closest approach equals
    *max_elevation_deg*.  With θ(t) = ω (t − t_closest):

        cos γ = cos β · cos θ
        d     = sqrt(R² + r² − 2 R r cos γ)
        sin ε = (r cos γ − R) / d
        ḋ     = R r ω cos β sin θ / d
        d̈     = (R r ω² cos β cos θ − ḋ²) / d
        f_D   = −(f_c / c) ḋ ,   f_D' = −(f_c / c) d̈

    Earth rotation is ignored (assumption A1).
    """

    def __init__(
        self,
        t_closest_s: float,
        max_elevation_deg: float = 90.0,
        h_km: float = DEFAULT_ALTITUDE,
        f_c: float = DEFAULT_FC,
        min_elevation_deg: float = DEFAULT_MIN_ELEVATION,
    ) -> None:
        self.t_closest = float(t_closest_s)
        self.max_elevation = float(max_elevation_deg)
        self.h_km = float(h_km)
        self.f_c = float(f_c)
        self.min_elevation = float(min_elevation_deg)
        self.r = (R_E + h_km) * 1e3                 # m
        self.R = R_E * 1e3                          # m
        self.omega = orbital_speed(h_km) / self.r   # rad/s
        self.cos_beta = math.cos(central_angle_for_elevation(min(max_elevation_deg, 89.999), h_km))
        # half-duration above the minimum elevation
        g_min = central_angle_for_elevation(min_elevation_deg, h_km)
        c = math.cos(g_min) / self.cos_beta
        self.half_duration = math.acos(c) / self.omega if c < 1.0 else 0.0

    @property
    def t_rise(self) -> float:
        return self.t_closest - self.half_duration

    @property
    def t_set(self) -> float:
        return self.t_closest + self.half_duration

    def state(self, t: float | np.ndarray) -> dict[str, np.ndarray | float]:
        """Elevation (deg), range (km), Doppler (Hz) and Doppler rate (Hz/s)."""
        theta = self.omega * (np.asarray(t, dtype=float) - self.t_closest)
        cos_g = self.cos_beta * np.cos(theta)
        d = np.sqrt(self.R ** 2 + self.r ** 2 - 2 * self.R * self.r * cos_g)
        sin_e = (self.r * cos_g - self.R) / d
        elev = np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))
        d_dot = self.R * self.r * self.omega * self.cos_beta * np.sin(theta) / d
        d_ddot = (self.R * self.r * self.omega ** 2 * self.cos_beta * np.cos(theta) - d_dot ** 2) / d
        k = self.f_c / C
        return {
            "elevation_deg": elev,
            "range_km": d / 1e3,
            "doppler_hz": -k * d_dot,
            "doppler_rate_hz_s": -k * d_ddot,
        }

    def visible(self, t: float) -> bool:
        return self.t_rise <= t <= self.t_set


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
