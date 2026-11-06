"""PHY: noise scaling, analytic SER, receiver models, PER tables."""
import math

import numpy as np
import pytest

from phy.css import (css_demodulate, css_modulate, noise_std_per_component, payload_symbols,
                     reduced_rate)
from phy.receiver import drift_bins_at_packet_end, max_trackable_rate, simulate_packets
from phy.ser_montecarlo import PerTable, ser_awgn_analytic, simulate_ser_awgn, threshold_per


def test_noise_std_gives_requested_snr():
    rng = np.random.default_rng(0)
    s = noise_std_per_component(-10.0)
    n = s * (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000))
    snr = 10 * math.log10(1.0 / np.mean(np.abs(n) ** 2))
    assert abs(snr - (-10.0)) < 0.05


def test_modulate_demodulate_roundtrip():
    for sf in (7, 9, 12):
        for m in (0, 1, 2 ** sf - 1, 2 ** (sf - 1)):
            assert css_demodulate(css_modulate(m, sf), sf) == m


@pytest.mark.parametrize("sf,snr", [(7, -10.5), (9, -15.5), (10, -18.0)])
def test_monte_carlo_matches_analytic_ser(sf, snr):
    mc = simulate_ser_awgn(sf, snr, n_trials=4000, rng=np.random.default_rng(1))
    th = ser_awgn_analytic(sf, snr)
    # binomial 4-sigma band
    assert abs(mc - th) < 4 * math.sqrt(th * (1 - th) / 4000) + 1e-3


def test_analytic_ser_monotone_and_sf_gain():
    assert ser_awgn_analytic(9, -14) > ser_awgn_analytic(9, -12)
    # one SF step buys between 2 and 3.5 dB at SER = 1e-2
    def snr_at(sf, target=1e-2):
        lo, hi = -40.0, 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ser_awgn_analytic(sf, mid) > target else (lo, mid)
        return hi
    gap = snr_at(9) - snr_at(10)
    assert 2.0 < gap < 3.5


def test_reduced_rate_symbols():
    assert reduced_rate(0, 7) and reduced_rate(7, 7)
    assert not reduced_rate(8, 7)
    assert reduced_rate(20, 12)          # DE = 1 for SF12


def test_receivers_noise_free_no_drift_are_error_free():
    for rx in ("preamble", "tracking"):
        r = simulate_packets(10, 30.0, 0.0, 50, receiver=rx, rng=np.random.default_rng(0))
        assert r.per == 0.0


def test_preamble_receiver_breaks_sf12_under_leo_rate_tracking_does_not():
    rng = np.random.default_rng(2)
    pre = simulate_packets(12, -10.0, 150.0, 40, receiver="preamble", rng=rng)
    trk = simulate_packets(12, -10.0, 150.0, 40, receiver="tracking", rng=rng)
    assert pre.per > 0.9
    assert trk.per < 0.1


def test_drift_helpers():
    assert max_trackable_rate(12) == pytest.approx(125e3 ** 2 / (2 * 4 ** 12))
    assert drift_bins_at_packet_end(12, 250.0) > drift_bins_at_packet_end(7, 250.0) * 50


def test_threshold_per():
    assert threshold_per(7, -7.0) == 0.0 and threshold_per(7, -8.0) == 1.0


def _toy_table():
    snr_rel = np.array([-2.0, 0.0, 2.0])
    rates = np.array([0.0, 100.0])
    per = np.zeros((6, 3, 2))
    per[:, 0, :] = 1.0
    per[:, 1, 0] = 0.5
    per[:, 1, 1] = 0.7
    per[:, 2, 1] = 0.2
    return PerTable({"receiver": "toy", "sf": np.arange(7, 13), "snr_rel_db": snr_rel,
                     "doppler_rate_hz_s": rates, "per": per})


def test_per_table_bilinear_and_clipping():
    t = _toy_table()
    assert t(7, -7.5, 0.0) == pytest.approx(0.5)            # threshold of SF7 = rel 0
    assert t(7, -6.5, 0.0) == pytest.approx(0.25)
    assert t(7, -7.5, 50.0) == pytest.approx(0.6)
    assert t(7, -30.0, 0.0) == pytest.approx(1.0)           # clipped to the edge
    assert t(7, -7.5, -100.0) == pytest.approx(0.7)         # |rate|


def test_per_table_min_snr():
    t = _toy_table()
    v = t.min_snr(7, 0.0, 0.25)
    assert t(7, v, 0.0) <= 0.25 + 1e-6
    assert math.isinf(t.min_snr(7, 100.0, 0.01))


def test_payload_symbols_counts():
    assert payload_symbols(20, 7) == 8 + math.ceil((160 - 28 + 28 + 16) / 28) * 5
