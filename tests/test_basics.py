"""Tests de base : filet de sécurité avant toute modification du code.

Lancer :  python -m pytest -q
Ces tests ne valident PAS la science du projet, seulement des valeurs de référence connues.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phy.css import time_on_air, fft_bin_width, symbol_duration, bit_rate, per_from_ser
from channel.satellite import slant_range, orbital_speed, fspl, simulate_pass


def test_toa_sf7_20_bytes():
    # Valeur de référence du cahier des charges : ~56.6 ms
    assert abs(time_on_air(20, 7) - 0.0566) < 0.0005


def test_toa_sf12_20_bytes():
    # SF12, DE=1 : ~1.32 s (valeur utilisée dans le README)
    assert abs(time_on_air(20, 12) - 1.319) < 0.01


def test_toa_doubles_roughly_per_sf_step():
    for sf in range(7, 12):
        ratio = time_on_air(20, sf + 1) / time_on_air(20, sf)
        assert 1.7 < ratio < 2.4


def test_fft_bin_width():
    assert abs(fft_bin_width(7) - 976.5625) < 1e-6
    assert abs(fft_bin_width(12) - 30.517578125) < 1e-6


def test_symbol_duration_and_bitrate():
    assert abs(symbol_duration(7) - 1.024e-3) < 1e-9
    assert abs(bit_rate(7) - 5468.75) < 1.0  # SF7, CR=1


def test_per_from_ser_bounds():
    assert per_from_ser(0.0, 50) == 0.0
    assert per_from_ser(1.0, 50) == 1.0
    assert 0.0 < per_from_ser(0.01, 50) < 1.0


def test_slant_range_zenith_equals_altitude():
    assert abs(slant_range(90.0, 600.0) - 600.0) < 1e-6


def test_slant_range_decreases_with_elevation():
    assert slant_range(10.0) > slant_range(45.0) > slant_range(90.0)


def test_orbital_speed_600km():
    assert abs(orbital_speed(600.0) / 1000.0 - 7.56) < 0.05


def test_fspl_reference():
    # 1000 km à 868 MHz : 20log10(1000)+20log10(868)+32.45
    expected = 60 + 20 * math.log10(868) + 32.45
    assert abs(fspl(1000.0, 868.0) - expected) < 1e-9


def test_pass_order_of_magnitude():
    p = simulate_pass()
    # Les clés exactes dépendent de l'implémentation : on ne teste que ce qui existe.
    keys = set(p.keys())
    assert len(keys) > 0
    n = len(next(iter(p.values())))
    # Passe au-dessus de 10° : de l'ordre de 8 à 12 minutes (dt = 1 s)
    assert 400 < n < 800
