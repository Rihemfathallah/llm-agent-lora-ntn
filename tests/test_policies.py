"""Baselines: ADR rule, oracle, rules controller, ML features."""
from collections import deque

import numpy as np

from netsim.config import load_config
from netsim.simulator import NetworkSimulator
from policies.adr import adr_step
from policies.fixed import FixedController
from policies.ml_classifier import FEATURE_NAMES, decode, device_features, encode
from policies.oracle import OracleController
from policies.rules import RulesController

THR = {"phy": {"receiver": "threshold"}, "network": {"n_devices": 60}}


def test_adr_step_examples():
    h = deque([0.0] * 20)
    # margin = 0 - (-20) - 10 = 10 -> 3 steps: SF12 -> SF9
    assert adr_step(h, 12, 14) == (9, 14)
    # margin = 10 - (-7.5) - 10 = 7.5 -> 2 steps; at SF7 they lower the power
    assert adr_step(deque([10.0] * 20), 7, 14) == (7, 8)
    # negative margin raises power
    assert adr_step(deque([-20.0] * 20), 7, 5) == (7, 14)
    # not enough history
    assert adr_step(deque([0.0] * 5), 12, 14) == (12, 14)


def test_fixed_controller_validation():
    import pytest
    with pytest.raises(ValueError):
        FixedController(13)


def test_oracle_never_worse_link_level_than_fixed():
    cfg = load_config("s1_jamming", overrides=THR)
    mo = NetworkSimulator(cfg, 0, OracleController()).run()
    mf = NetworkSimulator(cfg, 0, FixedController(7)).run()
    assert mo["per_failed"] <= mf["per_failed"]
    assert mo["energy_per_delivered_mj"] < 3 * mf["energy_per_delivered_mj"]


def test_oracle_waits_when_no_gateway():
    cfg = load_config("s3_pass_end", overrides=THR)
    m = NetworkSimulator(cfg, 0, OracleController()).run()
    assert m["no_gateway"] == 0 and m["withheld"] > 0


def test_rules_avoid_jammed_channel():
    cfg = load_config("s1_jamming", overrides=THR)
    rc = RulesController()
    NetworkSimulator(cfg, 0, rc).run()
    assert any(a["rule"] == "avoid_jammed_channel" and 868.1 not in a["fields"]["channels"]
               for a in rc.actions if a["t"] >= 3600)


def test_ml_label_codec_and_features():
    for sf in range(7, 13):
        for p in (2, 5, 8, 11, 14):
            assert decode(encode(sf, p)) == (sf, p)
    cfg = load_config("s1_jamming", overrides=THR)
    sim = NetworkSimulator(cfg, 0, FixedController(7))
    x = device_features(sim, sim.devices[0], 0.0)
    assert x.shape == (len(FEATURE_NAMES),) and np.isnan(x[0])


def test_adr_refers_snr_to_its_base_power():
    from types import SimpleNamespace
    from policies.adr import ADRController
    cfg = load_config("s1_jamming", overrides=THR)
    ctrl = ADRController()
    sim = NetworkSimulator(cfg, 0, ctrl)
    dev = sim.devices[0]
    base_pw = dev.base.tx_power_dbm
    ctrl.hist[0].clear()
    rec = SimpleNamespace(snr_db=5.0, tx_power_dbm=base_pw + 6)
    sim.set_base = lambda d, p: None
    ctrl.on_uplink(sim, dev, rec)
    assert ctrl.hist[0][-1] == -1.0
