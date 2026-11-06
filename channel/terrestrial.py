"""Terrestrial channel propagation models for LoRa (Okumura-Hata and log-normal shadowing).

Implements:
  - Okumura-Hata urban and rural path loss models (150-1500 MHz)
  - Log-normal shadow fading sampling
  - Link SNR computation for terrestrial LoRa gateway
"""

import math
from typing import Literal
import numpy as np

from channel.link_budget import (
    hata_urban_loss,
    hata_rural_loss,
    hata_correction_factor,
    terrestrial_snr,
)

__all__ = [
    "hata_urban_loss",
    "hata_rural_loss",
    "hata_correction_factor",
    "terrestrial_snr",
    "terrestrial_path_loss",
    "sample_log_normal_shadowing",
]


def sample_log_normal_shadowing(
    std_db: float = 6.0,
    size: int | tuple[int, ...] = 1,
    rng: np.random.Generator | None = None,
) -> float | np.ndarray:
    """Sample zero-mean log-normal shadow fading in dB.

    Parameters
    ----------
    std_db : float
        Standard deviation in dB (typically 6 dB for rural/suburban, 8-10 dB for urban).
    size : int or tuple of int
        Number of samples to draw.
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility.

    Returns
    -------
    float or np.ndarray
        Shadow fading sample(s) in dB.
    """
    if rng is None:
        val = np.random.normal(0.0, std_db, size=size)
    else:
        val = rng.normal(0.0, std_db, size=size)
    if size == 1 and isinstance(val, np.ndarray):
        return float(val.item())
    return val


def terrestrial_path_loss(
    distance_km: float,
    frequency_mhz: float = 868.1,
    h_tx_m: float = 1.5,
    h_rx_m: float = 30.0,
    environment: Literal["rural", "urban"] = "rural",
) -> float:
    """Compute Okumura-Hata terrestrial path loss in dB.

    Parameters
    ----------
    distance_km : float
        Link distance in kilometers (valid range 1 to 20 km).
    frequency_mhz : float
        Carrier frequency in MHz (default 868.1 MHz).
    h_tx_m : float
        Transmitter (end device) height in meters (default 1.5 m).
    h_rx_m : float
        Receiver (gateway) antenna height in meters (default 30 m, the lower
        bound of the Hata validity range).
    environment : {'rural', 'urban'}
        Propagation environment type.

    Returns
    -------
    float
        Path loss in dB.
    """
    # Signature of the Hata helpers: (f_mhz, h_b, h_m, d_km).  The first
    # version passed (f, d, h_m, h_b), i.e. the distance as the gateway height.
    if environment == "rural":
        return hata_rural_loss(frequency_mhz, h_rx_m, h_tx_m, distance_km)
    elif environment == "urban":
        return hata_urban_loss(frequency_mhz, h_rx_m, h_tx_m, distance_km)
    else:
        raise ValueError(f"Unknown environment: {environment}. Use 'rural' or 'urban'.")
