"""ML classifier baseline: HistGradientBoostingClassifier on oracle labels.

Section 3.5 phase 4.  The classifier runs on the network side, like ADR:
after each delivered uplink it predicts the (SF, power) the device should use
for its next reading, and the decision reaches the device in the class-A
downlink.  Link and channels stay those of the deployment.

Features (network-server view, same information as the agent, pitfall 3):
    last 5 SNRs of the device (NaN-padded; HGB handles NaN natively),
    device loss rate over its last 10 uplinks (frame-counter gaps),
    current SF, current power, link (0 terrestrial / 1 satellite),
    satellite elevation now (NaN if not visible), |Doppler rate| now,
    anomaly flag (detector alert in the last 10 min).

Labels: the oracle's lowest-energy (SF, power) with PER ≤ 1 % on the true
channel of the device's *next* reading (link fixed to the device's link,
channel = the one it will hop to).  No feasible setting → (SF12, 14 dBm).

Training data come from simulations on ``train_seeds`` (disjoint from the
evaluation seeds) of every scenario family, with a behaviour policy that
follows ADR and explores a random (SF, power) on 30 % of the uplinks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from netsim.devices import TxParams
from policies.adr import ADRController
from policies.base import Controller
from policies.oracle import OracleController

K_SNR = 5
POWERS = [2, 5, 8, 11, 14]
FEATURE_NAMES = [f"snr_{i}" for i in range(K_SNR)] + [
    "loss_rate", "sf", "power", "link_sat", "elevation", "doppler_rate", "anomaly_flag"]


def device_features(sim: Any, dev: Any, now: float) -> np.ndarray:
    hist = sim.ns.dev_history.get(dev.device_id, [])
    snrs = [r.snr_db for r in hist[-K_SNR:]]
    snrs = [math.nan] * (K_SNR - len(snrs)) + snrs
    last = hist[-10:]
    lost = sum(r.lost_before for r in last)
    loss = lost / (lost + len(last)) if last else math.nan
    st = sim.satellite_state(now) if sim.passes else {"visible": False}
    p = dev.effective_params(now)
    flag = 1.0 if any(a.time >= now - 600.0 for a in sim.detector.alerts) else 0.0
    return np.array(snrs + [
        loss, p.sf, p.tx_power_dbm, 1.0 if p.link == "satellite" else 0.0,
        st["elevation_deg"] if st["visible"] else math.nan,
        abs(st["doppler_rate_hz_s"]) if st["visible"] else 0.0,
        flag], dtype=float)


def encode(sf: int, power: int) -> int:
    return (sf - 7) * len(POWERS) + POWERS.index(power)


def decode(label: int) -> tuple[int, int]:
    return 7 + label // len(POWERS), POWERS[label % len(POWERS)]


class _Collector(ADRController):
    """Behaviour policy for data collection: ADR + 30 % random exploration."""
    name = "collector"

    def __init__(self, explore: float = 0.3) -> None:
        super().__init__()
        self.explore = explore
        self.X: list[np.ndarray] = []
        self.y: list[int] = []
        self.oracle = OracleController()

    def attach(self, sim: Any) -> None:
        super().attach(sim)
        self.oracle.setup(sim)
        self.rng = np.random.default_rng([sim.seed, 5555])

    def on_uplink(self, sim: Any, dev: Any, rec: Any) -> None:
        k_next = rec_reading_idx(sim, dev) + 1
        if k_next < len(dev.reading_times):
            t_next = float(dev.reading_times[k_next])
            ch_all = tuple(sim.channels)
            ch = ch_all[min(int(dev.u_channel[k_next] * len(ch_all)), len(ch_all) - 1)]
            link = dev.effective_params(sim.now).link
            link = dev.home_link if link == "wait" else link
            best = self.oracle.best(sim, dev, k_next, t_next, links=(link,), channels=(ch,))
            label = encode(12, 14) if best.link == "wait" else encode(best.sf, best.tx_power_dbm)
            self.X.append(device_features(sim, dev, sim.now))
            self.y.append(label)
        super().on_uplink(sim, dev, rec)
        if self.rng.random() < self.explore:
            sf = int(self.rng.integers(7, 13))
            pw = int(self.rng.choice(POWERS))
            sim.set_base(dev, dev.base.replace(sf=sf, tx_power_dbm=pw))


def rec_reading_idx(sim: Any, dev: Any) -> int:
    """Reading index of the packet the network server just received.

    Only used offline to build labels: the packet was appended to
    ``sim.packets`` just before ``on_uplink`` was called.
    """
    pkt = sim.packets[-1]
    assert pkt.device_id == dev.device_id
    return pkt.reading_idx


def collect_training_data(cfgs: list[dict[str, Any]], seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    from netsim.simulator import NetworkSimulator
    X, y = [], []
    for cfg in cfgs:
        for s in seeds:
            col = _Collector()
            NetworkSimulator(cfg, s, col).run()
            X.extend(col.X)
            y.extend(col.y)
    return np.asarray(X), np.asarray(y)


def train_model(X: np.ndarray, y: np.ndarray, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    clf.fit(X, y)
    return clf


class MLController(Controller):
    name = "ML-HGB"

    def __init__(self, model: Any) -> None:
        self.model = model
        self.adr = ADRController()          # only for the warm-start state
        self.changes = 0

    def attach(self, sim: Any) -> None:
        self.adr.attach(sim)                # same initial parameters as ADR

    def on_uplink(self, sim: Any, dev: Any, rec: Any) -> None:
        x = device_features(sim, dev, sim.now).reshape(1, -1)
        sf, pw = decode(int(self.model.predict(x)[0]))
        if (sf, pw) != (dev.base.sf, dev.base.tx_power_dbm):
            sim.set_base(dev, dev.base.replace(sf=sf, tx_power_dbm=pw))
            self.changes += 1

    def extra_metrics(self) -> dict[str, Any]:
        return {"ml_changes": self.changes}


def save_model(model: Any, path: str | Path) -> None:
    import joblib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    import joblib
    return joblib.load(path)
