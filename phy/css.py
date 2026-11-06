"""LoRa Chirp Spread Spectrum (CSS) physical layer.

Implements CSS modulation / demodulation, Time-on-Air calculation,
Doppler effects, and receiver sensitivity computation.

All formulas strictly follow the Semtech LoRa Modem Designer's Guide
(AN1200.13) and the project specification (Section 3.3).
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Demodulation SNR thresholds (dB) per spreading factor (commonly used values)
SNR_THRESHOLDS_DB: dict[int, float] = {
    7:  -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0,
}

# Default LoRa parameters
DEFAULT_BW = 125_000          # Hz  (125 kHz)
DEFAULT_CR = 1                # coding rate denominator offset, CR ∈ {1,2,3,4}
DEFAULT_CRC = 1               # CRC enabled
DEFAULT_IH = 0                # explicit header (IH = 0)
DEFAULT_N_PREAMBLE = 8        # preamble symbols


# ═══════════════════════════════════════════════════════════════════════════
# 1.  CSS Modulation
# ═══════════════════════════════════════════════════════════════════════════

def css_modulate(symbol: int, sf: int) -> np.ndarray:
    """Generate a CSS baseband chirp for a given symbol value.

    Sampled at f_s = BW (normalised, BW cancels out of the phase expression).

        x_m[n] = exp( j2π · ( n² / (2N) + (m/N − 1/2) · n ) )
        n = 0, …, N−1     with  N = 2^SF

    Parameters
    ----------
    symbol : int
        Symbol value *m* ∈ {0, …, N−1}.
    sf : int
        Spreading factor (7–12).

    Returns
    -------
    np.ndarray
        Complex baseband samples of length N = 2^SF.
    """
    N = 2 ** sf
    if not 0 <= symbol < N:
        raise ValueError(
            f"Symbol {symbol} out of range [0, {N - 1}] for SF{sf}"
        )
    n = np.arange(N, dtype=np.float64)
    phase = 2.0 * np.pi * (n ** 2 / (2 * N) + (symbol / N - 0.5) * n)
    return np.exp(1j * phase)


def css_base_chirp(sf: int) -> np.ndarray:
    """Generate the base (reference) up-chirp x_0[n] used for demodulation."""
    return css_modulate(0, sf)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  CSS Demodulation
# ═══════════════════════════════════════════════════════════════════════════

def css_demodulate(signal: np.ndarray, sf: int) -> int:
    """Demodulate a received CSS symbol.

    Multiply the received symbol by conj(x_0[n]), which leaves a pure tone
    at frequency m/N; take the FFT and pick the strongest bin.

    Parameters
    ----------
    signal : np.ndarray
        Received complex baseband samples (length N = 2^SF).
    sf : int
        Spreading factor (7–12).

    Returns
    -------
    int
        Detected symbol value *m* ∈ {0, …, N−1}.
    """
    ref = css_base_chirp(sf)
    dechirped = signal * np.conj(ref)
    spectrum = np.abs(np.fft.fft(dechirped))
    return int(np.argmax(spectrum))


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Doppler
# ═══════════════════════════════════════════════════════════════════════════

def apply_doppler(
    signal: np.ndarray,
    bw: float,
    f_d: float,
    f_d_rate: float = 0.0,
) -> np.ndarray:
    """Apply a (possibly drifting) Doppler frequency offset to a signal.

    Multiplies the samples by:

        exp( j2π · (f_D · t  +  ½ · f_D' · t²) )      with  t = n / BW

    Parameters
    ----------
    signal : np.ndarray
        Complex baseband samples.
    bw : float
        Bandwidth in Hz (= sampling rate f_s).
    f_d : float
        Static Doppler frequency offset (Hz).
    f_d_rate : float
        Doppler rate f_D' (Hz/s) — how fast the offset drifts.

    Returns
    -------
    np.ndarray
        Signal with Doppler applied.
    """
    n = np.arange(len(signal), dtype=np.float64)
    t = n / bw
    phase = 2.0 * np.pi * (f_d * t + 0.5 * f_d_rate * t ** 2)
    return signal * np.exp(1j * phase)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Time on Air  (Semtech AN1200.13)
# ═══════════════════════════════════════════════════════════════════════════

def symbol_duration(sf: int, bw: float = DEFAULT_BW) -> float:
    """Symbol duration  T_s = 2^SF / BW   (seconds)."""
    return (2 ** sf) / bw


def low_data_rate_optimise(sf: int, bw: float = DEFAULT_BW) -> int:
    """Low data-rate optimisation flag.

    DE = 1  when T_s > 16 ms  (SF11 and SF12 at 125 kHz), else 0.
    """
    return 1 if symbol_duration(sf, bw) > 16e-3 else 0


def time_on_air(
    payload_bytes: int,
    sf: int,
    bw: float = DEFAULT_BW,
    cr: int = DEFAULT_CR,
    crc: int = DEFAULT_CRC,
    ih: int = DEFAULT_IH,
    n_preamble: int = DEFAULT_N_PREAMBLE,
) -> float:
    """Compute LoRa Time on Air (seconds).

    Formulas (Semtech LoRa Modem Designer's Guide):

        T_preamble = (n_preamble + 4.25) · T_s

        n_payload  = 8 + max(
            ceil( (8·PL − 4·SF + 28 + 16·CRC − 20·IH)
                  / (4·(SF − 2·DE)) ) · (CR + 4),
            0 )

        ToA = T_preamble + n_payload · T_s

    Sanity check: SF7, 125 kHz, 20 bytes, CR = 1  →  ≈ 56.6 ms.

    Parameters
    ----------
    payload_bytes : int
        Number of application-layer payload bytes (PL).
    sf : int
        Spreading factor (7–12).
    bw : float
        Bandwidth in Hz.
    cr : int
        Coding rate denominator offset (1–4).  Actual rate = 4 / (4 + CR).
    crc : int
        1 if CRC enabled, 0 otherwise.
    ih : int
        1 if implicit header, 0 for explicit header.
    n_preamble : int
        Number of preamble symbols.

    Returns
    -------
    float
        Time on air in seconds.
    """
    t_s = symbol_duration(sf, bw)
    de = low_data_rate_optimise(sf, bw)

    t_preamble = (n_preamble + 4.25) * t_s

    numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * crc - 20 * ih
    denominator = 4 * (sf - 2 * de)
    n_payload = 8 + max(math.ceil(numerator / denominator) * (cr + 4), 0)

    return t_preamble + n_payload * t_s


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Bit rate
# ═══════════════════════════════════════════════════════════════════════════

def bit_rate(sf: int, bw: float = DEFAULT_BW, cr: int = DEFAULT_CR) -> float:
    """LoRa bit rate (bits / s).

        R_b = SF · (BW / 2^SF) · 4 / (4 + CR)
    """
    return sf * (bw / (2 ** sf)) * 4.0 / (4 + cr)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Receiver sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def receiver_sensitivity(
    sf: int, bw: float = DEFAULT_BW, nf_db: float = 6.0
) -> float:
    """Receiver sensitivity (dBm).

        S = −174 + 10·log10(BW) + NF + SNR_req

    Parameters
    ----------
    sf : int
        Spreading factor (7–12).
    bw : float
        Bandwidth in Hz.
    nf_db : float
        Receiver noise figure in dB.

    Returns
    -------
    float
        Sensitivity in dBm.
    """
    snr_req = SNR_THRESHOLDS_DB[sf]
    return -174.0 + 10.0 * math.log10(bw) + nf_db + snr_req


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Derived helpers
# ═══════════════════════════════════════════════════════════════════════════

def fft_bin_width(sf: int, bw: float = DEFAULT_BW) -> float:
    """FFT frequency resolution  BW / 2^SF  (Hz).

    ≈ 976 Hz at SF7, ≈ 30.5 Hz at SF12  (125 kHz BW).
    """
    return bw / (2 ** sf)


def per_from_ser(ser: float, n_symbols: int) -> float:
    """Packet error rate from symbol error rate (uncoded approximation).

        PER = 1 − (1 − SER)^n_symbols

    Only valid when symbol errors are independent (AWGN, no drift).  Under a
    Doppler *rate* the errors concentrate at the end of the packet, so the
    PER tables are computed packet by packet in :mod:`phy.receiver` instead.
    """
    return 1.0 - (1.0 - ser) ** n_symbols


def noise_std_per_component(snr_db: float) -> float:
    """Per-component (I or Q) noise standard deviation for unit-power chirps.

    The chirp samples have |x|² = 1, so the complex noise variance is
    σ_c² = 10^(−SNR/10), split equally between I and Q:

        σ = sqrt( 10^(−SNR/10) / 2 )

    (The first version of this project used σ = 10^(−SNR/20) per component,
    which doubles the noise power and shifts every SER curve by 3 dB.)
    """
    return math.sqrt(10.0 ** (-snr_db / 10.0) / 2.0)


N_HEADER_SYMBOLS = 8   # the first 8 payload-block symbols (explicit header)


def reduced_rate(symbol_index: int, sf: int, bw: float = DEFAULT_BW) -> bool:
    """True if payload symbol *symbol_index* carries only SF−2 bits.

    In LoRa the first block of 8 symbols (header block) is always sent at
    reduced rate (SF−2 bits, i.e. the two LSBs are not used), and the rest
    of the payload is reduced rate when low-data-rate optimisation is on
    (DE = 1, SF11/SF12 at 125 kHz).  A reduced-rate symbol tolerates a
    demodulation offset of up to ±2 FFT bins instead of ±0.5.
    """
    if symbol_index < N_HEADER_SYMBOLS:
        return True
    return low_data_rate_optimise(sf, bw) == 1


def payload_symbols(
    payload_bytes: int,
    sf: int,
    bw: float = DEFAULT_BW,
    cr: int = DEFAULT_CR,
    crc: int = DEFAULT_CRC,
    ih: int = DEFAULT_IH,
) -> int:
    """Number of payload symbols n_payload (for PER calculation)."""
    de = low_data_rate_optimise(sf, bw)
    numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * crc - 20 * ih
    denominator = 4 * (sf - 2 * de)
    return 8 + max(math.ceil(numerator / denominator) * (cr + 4), 0)
