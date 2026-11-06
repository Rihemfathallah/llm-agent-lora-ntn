"""Packet-level LoRa receiver models under AWGN and a Doppler *rate*.

Why this module exists
----------------------
The first version applied the Doppler drift inside a single symbol, starting
from zero at every symbol.  That only captures the chirp-rate mismatch
(a few hertz per symbol) and misses the dominant effect: the carrier keeps
drifting during the whole packet, while the receiver estimated the frequency
offset once, on the preamble.  For SF12 (≈1.3 s on air, 30.5 Hz FFT bins) the
drift accumulated by the end of the payload is many bins.

Model
-----
* The static Doppler shift is estimated on the preamble and removed.  The
  estimate refers to the centre of the preamble (n_preamble + 4.25 symbols).
* Payload symbol k starts at t_k = (n_preamble + 4.25 + k)·T_s; during the
  symbol the residual offset is  f(t) = f_res + f_D'·(t − t_ref).
* Each symbol is dechirped, FFT'd, and the strongest bin is taken.
* Reduced-rate symbols (header block, or DE = 1 for SF11/12) only carry
  SF−2 bits, so the decision is the bin rounded to a multiple of 4: they
  tolerate ±2 bins of offset instead of ±0.5.
* The tracking receiver starts from the frequency of the last preamble
  up-chirp (its loop already runs on the preamble), with no knowledge of the
  drift rate, and integrates the fractional offset measured on each symbol.
* With coding rate 4/5 the Hamming(5,4) code only *detects* errors, so a
  single wrong symbol loses the packet (CRC).  The packet error is therefore
  "at least one wrong symbol", computed packet by packet: no
  PER = 1 − (1 − SER)^n approximation, which would be wrong here because the
  drift concentrates errors at the end of the packet.

Two receivers bracket real hardware:

``"preamble"``  (no tracking)
    The offset is corrected once from the preamble.  Pessimistic bound:
    typical of a simple SDR implementation.

``"tracking"``  (decision-directed frequency tracking)
    After each symbol the fractional position of the FFT peak (parabolic
    interpolation) is fed back into the frequency correction of the next
    symbol (first-order loop).  The loop follows the drift as long as it
    moves by less than half a bin per symbol, i.e. f_D'·T_s < BW/(2·2^SF).
    Commercial chips track drift in some way; Doroshkin et al. (IEEE Access,
    2019) measured SF12 surviving zenith passes above 550 km, which a
    preamble-only receiver cannot reproduce.

Both are models, not a reproduction of the proprietary Semtech demodulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phy.css import (
    DEFAULT_BW,
    DEFAULT_CR,
    DEFAULT_N_PREAMBLE,
    css_base_chirp,
    noise_std_per_component,
    payload_symbols,
    reduced_rate,
)

RECEIVERS = ("preamble", "tracking")


@dataclass
class PacketSimResult:
    n_packets: int
    n_packet_errors: int
    n_symbols: int
    n_symbol_errors: int

    @property
    def per(self) -> float:
        return self.n_packet_errors / self.n_packets

    @property
    def ser(self) -> float:
        return self.n_symbol_errors / (self.n_packets * self.n_symbols)


def _fractional_offset(spec: np.ndarray, peak: np.ndarray) -> np.ndarray:
    """Fractional peak offset in bins (−0.5…0.5), Jacobsen estimator.

    δ = −Re[ (X[p+1] − X[p−1]) / (2·X[p] − X[p−1] − X[p+1]) ]

    computed on the complex spectrum *spec* (shape (B, N)) around the peak
    bins *peak* (shape (B,)), with circular neighbours.  Unlike parabolic
    interpolation on the magnitude, it is nearly unbiased for a rectangular
    window, which matters because the tracking loop integrates the estimate.
    """
    n = spec.shape[1]
    rows = np.arange(spec.shape[0])
    xm = spec[rows, (peak - 1) % n]
    x0 = spec[rows, peak]
    xp = spec[rows, (peak + 1) % n]
    denom = 2.0 * x0 - xm - xp
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(np.abs(denom) > 1e-12, -np.real((xp - xm) / denom), 0.0)
    return np.clip(delta, -0.5, 0.5)


def simulate_packets(
    sf: int,
    snr_db: float,
    doppler_rate_hz_s: float,
    n_packets: int,
    receiver: str = "tracking",
    payload_bytes: int = 20,
    bw: float = DEFAULT_BW,
    cr: int = DEFAULT_CR,
    n_preamble: int = DEFAULT_N_PREAMBLE,
    residual_cfo_hz: float = 0.0,
    rng: np.random.Generator | None = None,
    batch: int = 256,
) -> PacketSimResult:
    """Monte Carlo packet error simulation (AWGN + Doppler rate).

    Parameters
    ----------
    sf, snr_db, doppler_rate_hz_s
        Spreading factor, SNR in the 125 kHz band (dB), Doppler rate (Hz/s).
    n_packets
        Number of packets to simulate.
    receiver
        ``"preamble"`` or ``"tracking"`` (see module docstring).
    residual_cfo_hz
        Static frequency error left after the preamble estimate (Hz).
    """
    if receiver not in RECEIVERS:
        raise ValueError(f"receiver must be one of {RECEIVERS}")
    if rng is None:
        rng = np.random.default_rng(0)

    N = 2 ** sf
    t_s = N / bw
    bin_hz = bw / N
    n_sym = payload_symbols(payload_bytes, sf, bw, cr)
    sigma = noise_std_per_component(snr_db)
    base_conj = np.conj(css_base_chirp(sf))
    n = np.arange(N, dtype=np.float64)
    tau = n / bw                                     # time inside a symbol
    chirp_phase = 2.0 * np.pi * (n ** 2 / (2 * N) - 0.5 * n)   # x_0 phase
    # Reference time of the frequency estimate.  Preamble-only: average over
    # the preamble, i.e. its centre.  Tracking: the loop already runs on the
    # preamble up-chirps, so its estimate refers to the last up-chirp.
    if receiver == "preamble":
        t_ref = 0.5 * (n_preamble + 4.25) * t_s
    else:
        t_ref = (n_preamble - 0.5) * t_s
    rr = np.array([reduced_rate(k, sf, bw) for k in range(n_sym)])

    pkt_err_total = 0
    sym_err_total = 0
    done = 0
    while done < n_packets:
        b = min(batch, n_packets - done)
        err_pkt = np.zeros(b, dtype=bool)
        eps_hat = np.zeros(b)                        # tracker estimate (bins)
        for k in range(n_sym):
            # transmitted symbol: multiple of 4 on reduced-rate symbols
            if rr[k]:
                m = 4 * rng.integers(0, N // 4, size=b)
            else:
                m = rng.integers(0, N, size=b)
            t0 = (n_preamble + 4.25 + k) * t_s - t_ref
            f0 = residual_cfo_hz + doppler_rate_hz_s * t0     # Hz at symbol start
            drift_phase = 2.0 * np.pi * (f0 * tau + 0.5 * doppler_rate_hz_s * tau ** 2)
            # x_m[n] = exp(j(chirp_phase + 2π m n / N))
            phase = chirp_phase[None, :] + 2.0 * np.pi * np.outer(m, n) / N + drift_phase[None, :]
            rx = np.exp(1j * phase)
            rx += sigma * (rng.standard_normal((b, N)) + 1j * rng.standard_normal((b, N)))
            de = rx * base_conj[None, :]
            if receiver == "tracking":
                de *= np.exp(-2j * np.pi * np.outer(eps_hat, n) / N)
            spec = np.fft.fft(de, axis=1)
            peak = np.argmax(np.abs(spec), axis=1)
            if receiver == "tracking":
                eps_hat = eps_hat + _fractional_offset(spec, peak)
            if rr[k]:
                dec = (np.rint(peak / 4.0).astype(np.int64) % (N // 4)) * 4
            else:
                dec = peak
            wrong = dec != m
            sym_err_total += int(wrong.sum())
            err_pkt |= wrong
        pkt_err_total += int(err_pkt.sum())
        done += b

    return PacketSimResult(
        n_packets=n_packets,
        n_packet_errors=pkt_err_total,
        n_symbols=n_sym,
        n_symbol_errors=sym_err_total,
    )


def max_trackable_rate(sf: int, bw: float = DEFAULT_BW) -> float:
    """Doppler rate (Hz/s) at which the drift reaches half a bin per symbol.

    Noise-free limit of the first-order tracking loop:
    f_D' · T_s = BW / (2 · 2^SF)   →   f_D' = BW² / (2 · 4^SF).
    """
    return bw ** 2 / (2.0 * 4.0 ** sf)


def drift_bins_at_packet_end(
    sf: int,
    doppler_rate_hz_s: float,
    payload_bytes: int = 20,
    bw: float = DEFAULT_BW,
    cr: int = DEFAULT_CR,
    n_preamble: int = DEFAULT_N_PREAMBLE,
) -> float:
    """Offset (in FFT bins) reached at the end of the payload for the
    preamble-only receiver."""
    N = 2 ** sf
    t_s = N / bw
    n_sym = payload_symbols(payload_bytes, sf, bw, cr)
    t_end = (n_preamble + 4.25 + n_sym) * t_s - 0.5 * (n_preamble + 4.25) * t_s
    return abs(doppler_rate_hz_s) * t_end / (bw / N)
