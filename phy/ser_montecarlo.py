"""Monte Carlo SER / PER estimation for LoRa CSS in AWGN and Doppler channels.

PER from uncoded SER — simplification that ignores coding.

Provides:
  - simulate_ser_awgn      : SER vs SNR for a given SF (AWGN only)
  - simulate_ser_doppler   : SER vs SNR + Doppler rate for a given SF
  - generate_per_table     : Build 3-D PER(SNR, SF, Doppler_rate) lookup table
  - save_per_table / load_per_table : Persist lookup table as compressed .npz
  - lookup_per             : Interpolate PER from the table
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from phy.css import (
    DEFAULT_BW,
    SNR_THRESHOLDS_DB,
    apply_doppler,
    css_demodulate,
    css_modulate,
    payload_symbols,
    per_from_ser,
)

# ---------------------------------------------------------------------------
# 1.  SER in AWGN
# ---------------------------------------------------------------------------


def simulate_ser_awgn(
    sf: int,
    snr_db: float,
    n_trials: int = 1000,
    bw: float = DEFAULT_BW,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate symbol error rate via Monte Carlo in an AWGN channel.

    PER from uncoded SER — simplification that ignores coding.

    For each trial a random symbol m ∈ {0, …, 2^SF − 1} is modulated,
    contaminated with AWGN at the requested SNR, and demodulated.  The
    fraction of incorrect demodulations is returned as the SER.

    The CSS chirp has unit power (|x|² = 1 per sample), so
    noise_std = 10^(−SNR_dB / 20) per real / imaginary component.

    Parameters
    ----------
    sf : int
        Spreading factor (7–12).
    snr_db : float
        Signal-to-noise ratio in dB.
    n_trials : int
        Number of Monte Carlo trials (increase for smoother curves).
    bw : float
        Bandwidth in Hz (only used for consistency; the normalised
        modulator is BW-independent).
    rng : numpy.random.Generator | None
        Random number generator.  A new one seeded with 0 is created
        when *None*.

    Returns
    -------
    float
        Estimated SER ∈ [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = 2 ** sf
    noise_std = 10.0 ** (-snr_db / 20.0)

    errors = 0
    for _ in range(n_trials):
        m = int(rng.integers(0, N))
        chirp = css_modulate(m, sf)
        noise = rng.normal(0, noise_std, N) + 1j * rng.normal(0, noise_std, N)
        noisy = chirp + noise
        m_hat = css_demodulate(noisy, sf)
        if m_hat != m:
            errors += 1

    return errors / n_trials


# ---------------------------------------------------------------------------
# 2.  SER with Doppler
# ---------------------------------------------------------------------------


def simulate_ser_doppler(
    sf: int,
    snr_db: float,
    f_d_rate: float,
    n_trials: int = 1000,
    bw: float = DEFAULT_BW,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate SER under AWGN **plus** a Doppler frequency drift.

    PER from uncoded SER — simplification that ignores coding.

    Real receivers estimate and remove the static offset f_D from the
    preamble, so we model a residual error plus the Doppler rate f_D'.
    Therefore *f_d* is set to 0 and only *f_d_rate* is applied.

    Parameters
    ----------
    sf : int
        Spreading factor (7–12).
    snr_db : float
        Signal-to-noise ratio in dB.
    f_d_rate : float
        Doppler rate (Hz/s).
    n_trials : int
        Number of Monte Carlo trials.
    bw : float
        Bandwidth in Hz.
    rng : numpy.random.Generator | None
        Random number generator.

    Returns
    -------
    float
        Estimated SER ∈ [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = 2 ** sf
    noise_std = 10.0 ** (-snr_db / 20.0)

    errors = 0
    for _ in range(n_trials):
        m = int(rng.integers(0, N))
        chirp = css_modulate(m, sf)
        noise = rng.normal(0, noise_std, N) + 1j * rng.normal(0, noise_std, N)
        noisy = chirp + noise
        # Apply Doppler with f_d=0 (static offset removed by receiver)
        noisy = apply_doppler(noisy, bw, f_d=0.0, f_d_rate=f_d_rate)
        m_hat = css_demodulate(noisy, sf)
        if m_hat != m:
            errors += 1

    return errors / n_trials


# ---------------------------------------------------------------------------
# 3.  PER lookup table generation
# ---------------------------------------------------------------------------


def generate_per_table(
    snr_range: np.ndarray | None = None,
    doppler_rate_range: np.ndarray | None = None,
    sfs: list[int] | None = None,
    payload_bytes: int = 20,
    n_trials: int = 1000,
    bw: float = DEFAULT_BW,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a 3-D PER(SF, SNR, Doppler_rate) lookup table.

    PER from uncoded SER — simplification that ignores coding.

    For every (SF, SNR, f_D') combination the SER is estimated via
    :func:`simulate_ser_doppler` and converted to PER with
    :func:`per_from_ser`.

    Parameters
    ----------
    snr_range : array-like
        SNR values in dB.
    doppler_rate_range : array-like
        Doppler rate values in Hz/s.
    sfs : list[int]
        Spreading factors to evaluate.
    payload_bytes : int
        Payload size in bytes (used to compute number of symbols).
    n_trials : int
        Monte Carlo trials per point.
    bw : float
        Bandwidth in Hz.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``'snr_db'``, ``'doppler_rate_hz_s'``, ``'sf'``,
        ``'per'`` (3-D ndarray indexed [sf_idx, snr_idx, doppler_idx]),
        ``'payload_bytes'``, ``'n_trials'``, ``'bw'``, ``'seed'``.
    """
    if snr_range is None:
        snr_range = np.arange(-25, 1, 0.5)
    if doppler_rate_range is None:
        doppler_rate_range = np.array([0, 50, 100, 150, 200, 275, 400, 600, 1000], dtype=float)
    if sfs is None:
        sfs = [7, 8, 9, 10, 11, 12]

    snr_range = np.asarray(snr_range, dtype=float)
    doppler_rate_range = np.asarray(doppler_rate_range, dtype=float)
    sf_arr = np.asarray(sfs, dtype=int)

    rng = np.random.default_rng(seed)

    per = np.zeros((len(sf_arr), len(snr_range), len(doppler_rate_range)))

    total = len(sf_arr) * len(snr_range) * len(doppler_rate_range)
    done = 0

    for i_sf, sf in enumerate(sf_arr):
        n_sym = payload_symbols(payload_bytes, sf, bw)
        for i_snr, snr_db in enumerate(snr_range):
            for i_fd, f_d_rate in enumerate(doppler_rate_range):
                ser = simulate_ser_doppler(
                    sf, snr_db, f_d_rate, n_trials=n_trials, bw=bw, rng=rng,
                )
                per[i_sf, i_snr, i_fd] = per_from_ser(ser, n_sym)
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  [{done}/{total}] SF{sf}  SNR={snr_db:+.1f} dB  "
                          f"f_D'={f_d_rate:.0f} Hz/s  SER={ser:.4f}  "
                          f"PER={per[i_sf, i_snr, i_fd]:.4f}")

    return {
        "snr_db": snr_range,
        "doppler_rate_hz_s": doppler_rate_range,
        "sf": sf_arr,
        "per": per,
        "payload_bytes": payload_bytes,
        "n_trials": n_trials,
        "bw": bw,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# 4.  Save / Load
# ---------------------------------------------------------------------------

_DEFAULT_TABLE_DIR = Path(__file__).resolve().parent / "tables"


def save_per_table(table: dict[str, Any], path: str | Path | None = None) -> Path:
    """Save a PER lookup table as a compressed ``.npz`` file.

    Parameters
    ----------
    table : dict
        Table produced by :func:`generate_per_table`.
    path : str or Path, optional
        Destination file.  Defaults to ``phy/tables/per_lookup.npz``.

    Returns
    -------
    Path
        Resolved path of the saved file.
    """
    if path is None:
        path = _DEFAULT_TABLE_DIR / "per_lookup.npz"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(path),
        snr_db=table["snr_db"],
        doppler_rate_hz_s=table["doppler_rate_hz_s"],
        sf=table["sf"],
        per=table["per"],
        payload_bytes=np.array(table["payload_bytes"]),
        n_trials=np.array(table["n_trials"]),
        bw=np.array(table["bw"]),
        seed=np.array(table["seed"]),
    )
    return path.resolve()


def load_per_table(path: str | Path | None = None) -> dict[str, Any]:
    """Load a PER lookup table from a compressed ``.npz`` file.

    Parameters
    ----------
    path : str or Path, optional
        Source file.  Defaults to ``phy/tables/per_lookup.npz``.

    Returns
    -------
    dict
        Same structure as returned by :func:`generate_per_table`.
    """
    if path is None:
        path = _DEFAULT_TABLE_DIR / "per_lookup.npz"
    path = Path(path)

    data = np.load(str(path))
    return {
        "snr_db": data["snr_db"],
        "doppler_rate_hz_s": data["doppler_rate_hz_s"],
        "sf": data["sf"],
        "per": data["per"],
        "payload_bytes": int(data["payload_bytes"]),
        "n_trials": int(data["n_trials"]),
        "bw": float(data["bw"]),
        "seed": int(data["seed"]),
    }


# ---------------------------------------------------------------------------
# 5.  Interpolation lookup
# ---------------------------------------------------------------------------


def lookup_per(
    table: dict[str, Any],
    sf: int,
    snr_db: float,
    doppler_rate: float,
) -> float:
    """Interpolate PER from the pre-computed lookup table.

    Uses :class:`scipy.interpolate.RegularGridInterpolator` for
    trilinear interpolation across (SF, SNR, Doppler_rate).

    Parameters
    ----------
    table : dict
        Lookup table from :func:`generate_per_table` or
        :func:`load_per_table`.
    sf : int
        Spreading factor.
    snr_db : float
        SNR in dB.
    doppler_rate : float
        Doppler rate in Hz/s.

    Returns
    -------
    float
        Interpolated PER clipped to [0, 1].
    """
    sf_axis = table["sf"].astype(float)
    snr_axis = table["snr_db"]
    doppler_axis = table["doppler_rate_hz_s"]

    interp = RegularGridInterpolator(
        (sf_axis, snr_axis, doppler_axis),
        table["per"],
        method="linear",
        bounds_error=False,
        fill_value=None,  # extrapolate via nearest
    )

    point = np.array([[float(sf), float(snr_db), float(doppler_rate)]])
    per_val = float(interp(point)[0])
    return float(np.clip(per_val, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 6.  CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  LoRa CSS  —  Monte Carlo PER Lookup Table Generator")
    print("=" * 65)

    snr_range = np.arange(-25, 1, 0.5)
    doppler_rate_range = np.array(
        [0, 50, 100, 150, 200, 275, 400, 600, 1000], dtype=float
    )
    sfs = [7, 8, 9, 10, 11, 12]
    payload_bytes = 20
    n_trials = 1000
    seed = 42

    print(f"\nSFs           : {sfs}")
    print(f"SNR range     : {snr_range[0]:.1f} … {snr_range[-1]:.1f} dB  "
          f"({len(snr_range)} points)")
    print(f"Doppler rates : {doppler_rate_range} Hz/s")
    print(f"Payload       : {payload_bytes} bytes")
    print(f"Trials        : {n_trials}")
    print(f"Seed          : {seed}\n")

    table = generate_per_table(
        snr_range=snr_range,
        doppler_rate_range=doppler_rate_range,
        sfs=sfs,
        payload_bytes=payload_bytes,
        n_trials=n_trials,
        seed=seed,
    )

    out_path = save_per_table(table)
    print(f"\nTable saved to: {out_path}\n")

    # ------------------------------------------------------------------
    # Summary: SER at threshold SNR for each SF (Doppler = 0)
    # ------------------------------------------------------------------
    print("-" * 65)
    print("  SER sanity check at demodulation threshold (Doppler = 0)")
    print("-" * 65)
    rng_check = np.random.default_rng(seed)
    for sf in sfs:
        thr = SNR_THRESHOLDS_DB[sf]
        ser = simulate_ser_awgn(sf, thr, n_trials=500, rng=rng_check)
        print(f"  SF{sf:2d}   threshold = {thr:+6.1f} dB   SER = {ser:.4f}")
    print("-" * 65)
