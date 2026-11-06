"""Network simulator: fairness, duty cycle, collisions, control plane, metrics."""
import math

import numpy as np
import pytest

from netsim.anomalies import AnomalyEvent, AnomalySchedule, DetectorAlert, RollingZScoreDetector, evaluate_detector
from netsim.config import load_config
from netsim.devices import TxParams, create_devices
from netsim.energy import tx_current_ma, tx_energy_joules
from netsim.simulator import NetworkSimulator, recovery_time
from policies.adr import ADRController
from policies.fixed import FixedSF7, FixedSF12MaxPower

THR = {"phy": {"receiver": "threshold"}}


def small(scen=None, **over):
    o = {"network": {"n_devices": 60}, **THR}
    o.update(over)
    return load_config(*( [scen] if scen else []), overrides=o)


def test_energy_model_matches_datasheet_points():
    assert tx_current_ma(7) == pytest.approx(20.0)
    assert tx_current_ma(13) == pytest.approx(29.0)
    assert tx_energy_joules(13, 1.0) == pytest.approx(3.3 * 0.029)
    with pytest.raises(ValueError):
        tx_current_ma(20)


def test_common_random_numbers_identical_across_policies():
    cfg = small("s1_jamming")
    a, b = create_devices(cfg, 3), create_devices(cfg, 3)
    for da, db in zip(a, b):
        assert np.array_equal(da.reading_times, db.reading_times)
        assert np.array_equal(da.u_per, db.u_per) and da.terr_gain_db == db.terr_gain_db


def test_same_seed_same_result_and_readings_policy_independent():
    cfg = small("s1_jamming")
    m1 = NetworkSimulator(cfg, 1, FixedSF7()).run()
    m2 = NetworkSimulator(cfg, 1, FixedSF7()).run()
    m3 = NetworkSimulator(cfg, 1, FixedSF12MaxPower()).run()
    same = lambda a, b: a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b))
    assert m1.keys() == m2.keys() and all(same(m1[k], m2[k]) for k in m1)
    assert m1["readings"] == m3["readings"]


def test_duty_cycle_respected_per_device():
    cfg = small("s5_traffic_surge")
    sim = NetworkSimulator(cfg, 0, FixedSF12MaxPower())
    sim.run()
    by_dev = {}
    for p in sorted(sim.packets, key=lambda p: p.start):
        if p.device_id in by_dev:
            prev = by_dev[p.device_id]
            assert p.start >= prev.start + 100 * prev.toa - 1e-9
        by_dev[p.device_id] = p
    assert sim.summary()["dropped_duty"] > 0


def test_collision_and_capture_rule():
    cfg = small(**{"network": {"n_devices": 2, "duty_cycle": 0.01}})
    sim = NetworkSimulator(cfg, 0, FixedSF7())
    from netsim.simulator import Packet
    a = Packet(1, 0, 0, 1, 7, 14, 868.1, "terrestrial", 0.0, 0.1)
    b = Packet(2, 1, 0, 1, 7, 14, 868.1, "terrestrial", 0.05, 0.1)
    sim.devices[0].z_fast[0] = 0.0
    sim.devices[1].z_fast[0] = 0.0
    sim.devices[0].terr_gain_db = -100.0
    sim.devices[1].terr_gain_db = -100.0 + 3.0       # 3 dB apart: both lost
    sim._on_tx_start(a)
    sim._on_tx_start(b)
    assert a.collided and b.collided
    c = Packet(3, 0, 1, 2, 7, 14, 868.1, "terrestrial", 10.0, 0.1)
    d = Packet(4, 1, 1, 2, 7, 14, 868.1, "terrestrial", 10.05, 0.1)
    sim.devices[1].terr_gain_db = -100.0 + 8.0       # 8 dB stronger: captures
    sim.devices[0].z_fast[1] = 0.0
    sim.devices[1].z_fast[1] = 0.0
    sim._active["terrestrial"].clear()
    sim._on_tx_start(c)
    sim._on_tx_start(d)
    assert c.collided and not d.collided
    e = Packet(5, 0, 2, 3, 8, 14, 868.1, "terrestrial", 10.06, 0.1)   # other SF: orthogonal
    sim._on_tx_start(e)
    assert not e.collided


def test_jamming_penalty_formula():
    s = AnomalySchedule([AnomalyEvent("jamming", 0, 10, {"channel_mhz": 868.1, "interference_db": 15.0})])
    assert s.snr_penalty_db(5, 868.1) == pytest.approx(10 * math.log10(1 + 10 ** 1.5))
    assert s.snr_penalty_db(5, 868.3) == 0.0 and s.snr_penalty_db(11, 868.1) == 0.0


def test_override_only_after_delivered_uplink_or_broadcast():
    cfg = small()
    sim = NetworkSimulator(cfg, 0, FixedSF7())
    dev = sim.devices[0]
    sim.issue_override([0], {"sf": 9}, 600.0)
    assert dev.effective_params(0.0).sf == 7          # not delivered yet
    sim.now = 100.0
    sim._deliver_pending(dev)
    assert dev.effective_params(100.0).sf == 9
    assert dev.effective_params(800.0).sf == 7        # lease expired


def test_gateway_outage_blocks_terrestrial():
    cfg = small("s4_gateway_outage")
    sim = NetworkSimulator(cfg, 0, FixedSF7())
    sim.run()
    during = [p for p in sim.packets if p.link == "terrestrial" and 3600 <= p.start < 5400]
    assert during and all(not p.delivered for p in during)


def test_satellite_packets_only_during_pass():
    cfg = small("s3_pass_end")
    sim = NetworkSimulator(cfg, 0, FixedSF7())
    sim.run()
    sat = [p for p in sim.packets if p.link == "satellite"]
    p = sim.passes[0]
    assert all(not q.delivered for q in sat if q.start > p.t_set or q.end < p.t_rise)


def test_adr_warm_start_is_a_fixed_point_and_lowers_sf():
    cfg = small()
    sim = NetworkSimulator(cfg, 0, ADRController())
    sfs = [d.base.sf for d in sim.devices]
    assert min(sfs) == 7 and np.mean(sfs) < 10


def test_recovery_time_definition():
    cfg = load_config()
    t = np.arange(0, 7200, 600.0)
    sent = np.full(len(t), 100.0)
    deliv = np.full(len(t), 95.0)
    ts = {"t_start": t, "sent": sent, "delivered": deliv.copy()}
    assert recovery_time(ts, 3600, cfg, 4200) == 0.0
    ts["delivered"][6] = 50.0           # bin [3600, 4200) below
    assert recovery_time(ts, 3600, cfg, 4200) == 600.0
    ts["delivered"][6:] = 50.0          # never recovers
    assert math.isnan(recovery_time(ts, 3600, cfg, 4200))


def test_detector_rising_edge_and_persistence():
    det = RollingZScoreDetector(z_threshold=3.0, window=20, min_samples=5, persistence=2)
    rng = np.random.default_rng(0)
    for i in range(20):
        det.observe(60.0 * i, {"noise_floor_dbm": {868.1: -117 + 0.3 * rng.standard_normal()}})
    assert not det.alerts
    det.observe(1300, {"noise_floor_dbm": {868.1: -100.0}})
    assert not det.alerts                         # persistence 2
    det.observe(1360, {"noise_floor_dbm": {868.1: -100.0}})
    det.observe(1420, {"noise_floor_dbm": {868.1: -100.0}})
    assert len(det.alerts) == 1 and det.alerts[0].kind == "noise_rise"


def test_detector_precision_recall():
    sch = AnomalySchedule([AnomalyEvent("jamming", 100, 200, {}), AnomalyEvent("deep_fade", 1000, 1100, {})])
    al = [DetectorAlert(150, "noise_rise", "jamming", 868.1, [], 5, 1),
          DetectorAlert(5000, "noise_rise", "jamming", 868.1, [], 5, 1)]
    r = evaluate_detector(al, sch, tolerance_s=60)
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["detection_delay_s"] == 50
