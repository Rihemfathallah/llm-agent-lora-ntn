"""Monte Carlo SER / PER for LoRa CSS, analytic reference, and PER lookup tables.

Provides
--------
simulate_ser_awgn        SER vs SNR (AWGN), vectorised Monte Carlo
ser_awgn_analytic        exact SER of noncoherent orthogonal 2^SF-ary signalling
simulate_ser_doppler     SER/PER of a full packet under a Doppler rate (phy.receiver)
generate_per_table       PER(SF, SNR, |Doppler rate|) for one receiver model
save_per_table / load_per_table
PerTable                 fast bilinear lookup used by the network simulator

PER is computed packet by packet (a packet is lost if any symbol is wrong),
see :mod:`phy.receiver` for why the PER = 1 − (1 − SER)^n shortcut is not used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import integrate, special

from phy.css import (
    DEFAULT_BW,
    SNR_THRESHOLDS_DB,
    css_base_chirp,
    noise_std_per_component,
)
from phy.receiver import simulate_packets

_DEFAULT_TABLE_DIR = Path(__file__).resolve().parent / "tables"


# ---------------------------------------------------------------------------
# 1.  SER in AWGN
# ---------------------------------------------------------------------------


def simulate_ser_awgn(
    sf: int,
    snr_db: float,
    n_trials: int = 1000,
    bw: float = DEFAULT_BW,
    rng: np.random.Generator | None = None,
    batch: int = 512,
) -> float:
    """Monte Carlo symbol error rate in AWGN (full-rate symbols).

    Unit-power chirps, complex noise variance 10^(−SNR/10).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = 2 ** sf
    sigma = noise_std_per_component(snr_db)
    n = np.arange(N, dtype=np.float64)
    chirp_phase = 2.0 * np.pi * (n ** 2 / (2 * N) - 0.5 * n)
    base_conj = np.conj(css_base_chirp(sf))
    errors = 0
    done = 0
    while done < n_trials:
        b = min(batch, n_trials - done)
        m = rng.integers(0, N, size=b)
        x = np.exp(1j * (chirp_phase[None, :] + 2.0 * np.pi * np.outer(m, n) / N))
        x += sigma * (rng.standard_normal((b, N)) + 1j * rng.standard_normal((b, N)))
        peak = np.argmax(np.abs(np.fft.fft(x * base_conj[None, :], axis=1)), axis=1)
        errors += int((peak != m).sum())
        done += b
    return errors / n_trials


def ser_awgn_analytic(sf: int, snr_db: float) -> float:
    """Exact SER of noncoherent detection of 2^SF orthogonal tones.

    After dechirping and FFT, the correct bin is Rician (ν² = 2·N·SNR in units
    of the per-component noise variance) and the N−1 others are Rayleigh:

        P_c = ∫ f_Rice(r; ν) · (1 − e^{−r²/2})^{N−1} dr,   SER = 1 − P_c.
    """
    N = 2 ** sf
    nu = np.sqrt(2.0 * N * 10.0 ** (snr_db / 10.0))

    def integrand(r: float) -> float:
        # Rician pdf with σ = 1, written with i0e for numerical stability
        rice = r * np.exp(-0.5 * (r - nu) ** 2) * special.i0e(r * nu)
        others = np.exp((N - 1) * np.log1p(-np.exp(-0.5 * r * r) + 1e-300))
        return rice * others

    lo, hi = max(0.0, nu - 12.0), nu + 12.0
    pc, _ = integrate.quad(integrand, lo, hi, limit=200, points=[nu])
    return float(min(1.0, max(0.0, 1.0 - pc)))


# ---------------------------------------------------------------------------
# 2.  Packet-level simulation under a Doppler rate
# ---------------------------------------------------------------------------


def simulate_ser_doppler(
    sf: int,
    snr_db: float,
    f_d_rate: float,
    n_trials: int = 200,
    receiver: str = "tracking",
    payload_bytes: int = 20,
    bw: float = DEFAULT_BW,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """SER and PER of *n_trials* full packets under a Doppler rate.

    Returns ``{"ser": …, "per": …}``.  See :mod:`phy.receiver`.
    """
    res = simulate_packets(
        sf, snr_db, f_d_rate, n_trials, receiver=receiver,
        payload_bytes=payload_bytes, bw=bw, rng=rng,
    )
    return {"ser": res.ser, "per": res.per}


# ---------------------------------------------------------------------------
# 3.  PER lookup tables
# ---------------------------------------------------------------------------

DEFAULT_SNR_REL = np.arange(-8.0, 8.01, 1.0)        # dB relative to SNR_req(SF)
DEFAULT_RATES = np.array(
    [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000],
    dtype=float,
)


def generate_per_table(
    receiver: str = "tracking",
    snr_rel_db: np.ndarray | None = None,
    doppler_rates: np.ndarray | None = None,
    sfs: list[int] | None = None,
    payload_bytes: int = 20,
    n_packets: int = 400,
    bw: float = DEFAULT_BW,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build PER[sf, snr, rate] for one receiver model.

    The SNR axis is relative to each SF's demodulation threshold, so every SF
    gets the same resolution around its waterfall.  Absolute SNR =
    SNR_req(SF) + snr_rel.
    """
    snr_rel = np.asarray(DEFAULT_SNR_REL if snr_rel_db is None else snr_rel_db, float)
    rates = np.asarray(DEFAULT_RATES if doppler_rates is None else doppler_rates, float)
    sfs = list(range(7, 13)) if sfs is None else list(sfs)
    per = np.zeros((len(sfs), len(snr_rel), len(rates)))
    for i, sf in enumerate(sfs):
        # one independent stream per SF so tables can be regenerated per SF
        rng = np.random.default_rng([seed, sf, 0 if receiver == "preamble" else 1])
        for j, rel in enumerate(snr_rel):
            for k, rate in enumerate(rates):
                res = simulate_packets(
                    sf, SNR_THRESHOLDS_DB[sf] + rel, rate, n_packets,
                    receiver=receiver, payload_bytes=payload_bytes, bw=bw, rng=rng,
                )
                per[i, j, k] = res.per
        if verbose:
            print(f"  [{receiver}] SF{sf} done", flush=True)
    return {
        "receiver": receiver,
        "sf": np.asarray(sfs, int),
        "snr_rel_db": snr_rel,
        "doppler_rate_hz_s": rates,
        "per": per,
        "payload_bytes": payload_bytes,
        "n_packets": n_packets,
        "bw": bw,
        "seed": seed,
    }


def save_per_table(table: dict[str, Any], path: str | Path | None = None) -> Path:
    if path is None:
        path = _DEFAULT_TABLE_DIR / f"per_{table['receiver']}.npz"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        receiver=np.array(table["receiver"]),
        sf=table["sf"],
        snr_rel_db=table["snr_rel_db"],
        doppler_rate_hz_s=table["doppler_rate_hz_s"],
        per=table["per"],
        payload_bytes=np.array(table["payload_bytes"]),
        n_packets=np.array(table["n_packets"]),
        bw=np.array(table["bw"]),
        seed=np.array(table["seed"]),
    )
    return path.resolve()


def load_per_table(receiver: str = "tracking", path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = _DEFAULT_TABLE_DIR / f"per_{receiver}.npz"
    d = np.load(str(path))
    return {
        "receiver": str(d["receiver"]),
        "sf": d["sf"],
        "snr_rel_db": d["snr_rel_db"],
        "doppler_rate_hz_s": d["doppler_rate_hz_s"],
        "per": d["per"],
        "payload_bytes": int(d["payload_bytes"]),
        "n_packets": int(d["n_packets"]),
        "bw": float(d["bw"]),
        "seed": int(d["seed"]),
    }


class PerTable:
    """Bilinear PER lookup per SF (no interpolation *across* SFs).

    Outside the SNR axis: PER of the nearest edge (−8 dB → ~1, +8 dB → the
    drift-limited floor).  Above the largest rate: PER of the largest rate.
    """

    def __init__(self, table: dict[str, Any]) -> None:
        self.receiver = table["receiver"]
        self.sfs = [int(s) for s in table["sf"]]
        self.snr_rel = np.asarray(table["snr_rel_db"], float)
        self.rates = np.asarray(table["doppler_rate_hz_s"], float)
        self.per = np.asarray(table["per"], float)
        self._idx = {sf: i for i, sf in enumerate(self.sfs)}

    @classmethod
    def load(cls, receiver: str = "tracking", path: str | Path | None = None) -> "PerTable":
        return cls(load_per_table(receiver, path))

    def __call__(self, sf: int, snr_db: float, doppler_rate: float = 0.0) -> float:
        grid = self.per[self._idx[int(sf)]]
        x = float(np.clip(snr_db - SNR_THRESHOLDS_DB[int(sf)], self.snr_rel[0], self.snr_rel[-1]))
        y = float(np.clip(abs(doppler_rate), self.rates[0], self.rates[-1]))
        i = min(int(np.searchsorted(self.snr_rel, x, side="right")) - 1, len(self.snr_rel) - 2)
        j = min(int(np.searchsorted(self.rates, y, side="right")) - 1, len(self.rates) - 2)
        i, j = max(i, 0), max(j, 0)
        tx = (x - self.snr_rel[i]) / (self.snr_rel[i + 1] - self.snr_rel[i])
        ty = (y - self.rates[j]) / (self.rates[j + 1] - self.rates[j])
        v = (grid[i, j] * (1 - tx) * (1 - ty) + grid[i + 1, j] * tx * (1 - ty)
             + grid[i, j + 1] * (1 - tx) * ty + grid[i + 1, j + 1] * tx * ty)
        return float(min(1.0, max(0.0, v)))


    def min_snr(self, sf: int, doppler_rate: float, target: float = 0.01) -> float:
        """Smallest SNR (dB) with PER ≤ *target* at this |rate| (inf if none).

        Bisection on the bilinear lookup (PER is non-increasing in SNR up to
        Monte Carlo noise).  Cached on a 1 Hz/s rate grid.
        """
        key = (int(sf), round(abs(doppler_rate)), target)
        cache = self.__dict__.setdefault("_min_snr_cache", {})
        if key in cache:
            return cache[key]
        thr = SNR_THRESHOLDS_DB[int(sf)]
        lo, hi = thr + self.snr_rel[0], thr + self.snr_rel[-1]
        if self(sf, hi, doppler_rate) > target:
            cache[key] = float("inf")
            return cache[key]
        if self(sf, lo, doppler_rate) <= target:
            cache[key] = lo
            return lo
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self(sf, mid, doppler_rate) <= target:
                hi = mid
            else:
                lo = mid
        cache[key] = hi
        return hi


def threshold_per(sf: int, snr_db: float, doppler_rate: float = 0.0) -> float:
    """Hard 0/1 model: PER = 0 above SNR_req(SF), 1 below (Doppler ignored)."""
    return 0.0 if snr_db >= SNR_THRESHOLDS_DB[sf] else 1.0


# ---------------------------------------------------------------------------
# 4.  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate PER lookup tables")
    ap.add_argument("--receiver", choices=["preamble", "tracking"], required=True)
    ap.add_argument("--n-packets", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tab = generate_per_table(args.receiver, n_packets=args.n_packets, seed=args.seed)
    print("saved", save_per_table(tab))
