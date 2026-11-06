"""Channel: Hata, TR 38.811 values, closed-form pass geometry."""
import math

import numpy as np
import pytest

from channel.link_budget import hata_rural_loss, link_budget
from channel.satellite import (SatellitePass, clutter_loss, extra_loss_sample, los_probability,
                               shadow_fading_std, simulate_pass, slant_range)
from channel.terrestrial import terrestrial_path_loss


def test_terrestrial_path_loss_argument_order_fixed():
    assert terrestrial_path_loss(5.0, 868.1, 1.5, 30.0) == pytest.approx(
        hata_rural_loss(868.1, 30.0, 1.5, 5.0))
    assert terrestrial_path_loss(10.0) > terrestrial_path_loss(2.0)


def test_tr38811_table_values():
    # 3GPP TR 38.811 V15.1.0, tables 6.6.1-1 and 6.6.2-3 (S-band, suburban/rural)
    assert los_probability(10) == pytest.approx(0.782)
    assert los_probability(90) == pytest.approx(0.998)
    assert float(clutter_loss(10, los=False)) == pytest.approx(19.52)
    assert float(clutter_loss(30, los=True)) == 0.0
    assert float(shadow_fading_std(40, los=False)) == pytest.approx(10.25)
    assert float(shadow_fading_std(40, los=True)) == pytest.approx(0.92)


def test_extra_loss_sample_los_and_nlos():
    loss, los = extra_loss_sample(30.0, 0.0, 0.0)
    assert los and loss == 0.0
    loss, los = extra_loss_sample(30.0, 0.999, 0.0)
    assert not los and loss == pytest.approx(18.42)


def test_closed_form_pass_matches_numeric_pass():
    p = SatellitePass(0.0, 90.0)
    num = simulate_pass()
    i = int(np.argmax(num["elevation_deg"]))
    st = p.state(0.0)
    assert st["doppler_rate_hz_s"] == pytest.approx(num["doppler_rate_hz_s"][i], rel=1e-3)
    assert 2 * p.half_duration == pytest.approx(len(num["t_s"]) - 1, abs=2.0)
    assert float(p.state(p.t_set)["elevation_deg"]) == pytest.approx(10.0, abs=1e-6)


def test_pass_max_doppler_and_range():
    p = SatellitePass(0.0, 90.0)
    st = p.state(p.t_rise)
    assert 19_000 < st["doppler_hz"] < 20_500
    assert float(st["range_km"]) == pytest.approx(slant_range(10.0), rel=1e-6)


def test_lower_pass_is_shorter_with_lower_rate():
    hi, lo = SatellitePass(0.0, 90.0), SatellitePass(0.0, 40.0)
    assert lo.half_duration < hi.half_duration
    assert abs(lo.state(0.0)["doppler_rate_hz_s"]) < abs(hi.state(0.0)["doppler_rate_hz_s"])
    assert float(lo.state(0.0)["elevation_deg"]) == pytest.approx(40.0, abs=1e-6)


def test_link_budget_formula():
    nf = -174 + 10 * math.log10(125e3) + 6
    assert link_budget(14, 0, 0, 140) == pytest.approx(14 - 140 - nf)
