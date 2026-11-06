"""Event-driven hybrid terrestrial + LEO LoRa network simulator.

Section 3.5, phase 3, extended to the hybrid scenario of section 3.2.

Model summary (every simplification is listed in docs/ASSUMPTIONS.md)
--------------------------------------------------------------------
* Two receivers: a terrestrial gateway (Okumura–Hata rural + slow log-normal
  shadowing per device + per-packet fading) and a LEO satellite gateway
  (closed-form pass geometry, FSPL, TR 38.811 LOS/NLOS losses, Doppler rate).
* A packet is addressed to one gateway (its *link*); collisions are resolved
  per gateway with pure ALOHA: same channel, same SF and time overlap
  → both lost unless one is ≥ 6 dB stronger (capture).  SFs are orthogonal.
* The device hops at random among its enabled channels (LoRaWAN behaviour).
* Duty cycle: 1 % on sub-band g1 per device (the three default channels are
  all in g1), i.e. silent for 99·ToA after each transmission.  A reading that
  cannot be sent before the next one is dropped.
* Success = no collision, gateway available, and a Bernoulli draw against the
  PER lookup table PER(SF, SNR, |Doppler rate|).  The network simulator never
  simulates chirps.
* Control plane: network-side decisions (ADR, ML, agent, rules) reach a
  device only in the downlink that follows one of its *delivered* uplinks
  (LoRaWAN class A), or, during a satellite pass, through a broadcast
  satellite downlink (assumption).  The oracle is the exception: it decides
  per packet with perfect knowledge (upper bound).
* Network-server view: delivered uplinks (SNR, SF, channel), losses inferred
  from frame-counter gaps, per-channel noise floor measured every tick.
  Controllers and the detector only use this view.
"""

from __future__ import annotations

import bisect
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from channel.satellite import SatellitePass, extra_loss_sample
from netsim.anomalies import AnomalySchedule, RollingZScoreDetector, evaluate_detector
from netsim.devices import Device, TxParams, create_devices, terrestrial_noise_floor_dbm
from netsim.energy import tx_energy_joules
from phy.css import SNR_THRESHOLDS_DB, time_on_air
from phy.ser_montecarlo import PerTable, threshold_per

EU868_CHANNELS = [868.1, 868.3, 868.5]

# event priorities at equal time: TX_END before READING before TICK
_TX_END, _TX_START, _READING, _TICK = 0, 1, 2, 3


@dataclass
class Packet:
    pkt_id: int
    device_id: int
    reading_idx: int
    fcnt: int
    sf: int
    tx_power_dbm: int
    channel: float
    link: str
    start: float
    toa: float
    rx_power_dbm: float = 0.0
    snr_db: float = 0.0
    doppler_rate: float = 0.0
    elevation_deg: float | None = None
    energy_j: float = 0.0
    collided: bool = False
    no_gateway: bool = False
    per_failed: bool = False

    @property
    def end(self) -> float:
        return self.start + self.toa

    @property
    def delivered(self) -> bool:
        return not (self.collided or self.no_gateway or self.per_failed)


@dataclass
class NSRecord:
    """What the network server learns from one delivered uplink."""
    time: float
    device_id: int
    sf: int
    tx_power_dbm: int
    channel: float
    link: str
    snr_db: float
    lost_before: int           # frame-counter gap revealed by this uplink


class NetworkServer:
    """Network-server-side view of the network (no ground truth)."""

    def __init__(self, n_devices: int) -> None:
        self.records: list[NSRecord] = []
        self._times: list[float] = []
        self.last_fcnt: dict[int, int] = {}
        self.last_seen: dict[int, float] = {}
        self.last_rx_by_gateway: dict[str, float] = {}
        self.dev_history: dict[int, list[NSRecord]] = defaultdict(list)
        self.noise_floor: dict[float, list[tuple[float, float]]] = defaultdict(list)
        self.n_devices = n_devices

    def receive(self, pkt: Packet) -> NSRecord:
        last = self.last_fcnt.get(pkt.device_id, pkt.fcnt - 1)
        rec = NSRecord(pkt.end, pkt.device_id, pkt.sf, pkt.tx_power_dbm, pkt.channel,
                       pkt.link, pkt.snr_db, max(0, pkt.fcnt - last - 1))
        self.last_fcnt[pkt.device_id] = pkt.fcnt
        self.last_seen[pkt.device_id] = pkt.end
        self.last_rx_by_gateway[pkt.link] = pkt.end
        self.records.append(rec)
        self._times.append(rec.time)
        self.dev_history[pkt.device_id].append(rec)
        return rec

    def window(self, now: float, window_s: float) -> list[NSRecord]:
        i = bisect.bisect_left(self._times, now - window_s)
        return self.records[i:]

    def snr_history(self, device_id: int, k: int = 20) -> list[float]:
        return [r.snr_db for r in self.dev_history[device_id][-k:]]


class NetworkSimulator:
    """One simulation run = one config + one seed + one controller."""

    def __init__(self, cfg: dict[str, Any], seed: int, controller: Any,
                 per_lookup: Callable[[int, float, float], float] | None = None) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.controller = controller
        self.duration = float(cfg["duration_s"])
        self.payload = int(cfg["phy"]["payload_bytes"])
        self.bw = float(cfg["phy"]["bandwidth_hz"])
        self.channels = [float(c) for c in cfg["network"]["channels_mhz"]]
        self.capture_db = float(cfg["network"]["capture_threshold_db"])
        self.duty_off = (1.0 - cfg["network"]["duty_cycle"]) / cfg["network"]["duty_cycle"]
        self.interval = float(cfg["network"]["report_interval_s"])
        self.tick_s = float(cfg["detector"]["tick_s"])
        self.noise_floor = terrestrial_noise_floor_dbm(cfg)
        sat = cfg["satellite"]
        self.sat_noise_floor = -174.0 + 10.0 * math.log10(self.bw) + sat["noise_figure_db"]
        self.sat_enabled = bool(sat.get("enabled", True))
        self.min_elev = float(sat["min_elevation_deg"])
        self.passes: list[SatellitePass] = []
        if self.sat_enabled:
            for p in sat.get("passes", []):
                tmp = SatellitePass(0.0, p["max_elevation_deg"], sat["altitude_km"],
                                    sat["frequency_hz"], self.min_elev)
                tc = p["t_closest_s"] if "t_closest_s" in p else p["t_set_s"] - tmp.half_duration
                self.passes.append(SatellitePass(tc, p["max_elevation_deg"], sat["altitude_km"],
                                                 sat["frequency_hz"], self.min_elev))
        if per_lookup is None:
            rx = cfg["phy"]["receiver"]
            per_lookup = threshold_per if rx == "threshold" else PerTable.load(rx)
        self.per_lookup = per_lookup
        self.toa = {sf: time_on_air(self.payload, sf, self.bw) for sf in range(7, 13)}

        self.devices: list[Device] = create_devices(cfg, self.seed)
        terr_ids = [d.device_id for d in self.devices if d.home_link == "terrestrial"]
        self.anomalies = AnomalySchedule.from_config(cfg, terr_ids, self.seed)
        dcfg = cfg["detector"]
        self.detector = RollingZScoreDetector(dcfg["z_threshold"], dcfg["window"],
                                              dcfg["min_samples"], dcfg["silence_s"],
                                              dcfg.get("persistence", 2))
        self.ns = NetworkServer(len(self.devices))
        self.now = 0.0
        self._q: list[tuple] = []
        self._seq = 0
        self._pkt_id = 0
        self._active: dict[str, list[Packet]] = {"terrestrial": [], "satellite": []}
        self.packets: list[Packet] = []
        self.pending: dict[int, tuple[dict[str, Any], float, float]] = {}   # dev -> (fields, lease, issued)
        self.overrides_issued = 0
        self.overrides_delivered = 0
        self._noise_rng = np.random.default_rng([self.seed, 4242])
        self.tick_log: list[dict[str, Any]] = []
        self.controller.attach(self)

    # ── helpers ───────────────────────────────────────────────────────

    def _push(self, t: float, prio: int, kind: str, payload: Any) -> None:
        self._seq += 1
        heapq.heappush(self._q, (t, prio, self._seq, kind, payload))

    def satellite_state(self, t: float) -> dict[str, Any]:
        """Geometry of the pass covering *t* (or the next one)."""
        for p in self.passes:
            if p.visible(t):
                s = p.state(t)
                return {"visible": True, "elevation_deg": float(s["elevation_deg"]),
                        "range_km": float(s["range_km"]), "doppler_hz": float(s["doppler_hz"]),
                        "doppler_rate_hz_s": float(s["doppler_rate_hz_s"]),
                        "time_to_set_s": p.t_set - t, "pass": p}
        upcoming = [p for p in self.passes if p.t_rise > t]
        nxt = min(upcoming, key=lambda p: p.t_rise) if upcoming else None
        return {"visible": False, "elevation_deg": None, "doppler_hz": None,
                "doppler_rate_hz_s": None, "time_to_set_s": None,
                "next_rise_in_s": (nxt.t_rise - t) if nxt else None,
                "next_max_elevation_deg": nxt.max_elevation if nxt else None, "pass": None}

    def true_snr(self, dev: Device, k: int, link: str, power: float, channel: float,
                 t: float) -> tuple[float, float, float | None, float]:
        """(SNR dB, rx power dBm, elevation, |Doppler rate|) of reading *k* now.

        Used by the simulator and by the oracle (which is allowed to look).
        Returns SNR = −inf when the gateway cannot receive.
        """
        fade = self.anomalies.extra_loss_db(t, dev.device_id)
        if link == "terrestrial":
            if self.anomalies.gateway_down(t):
                return -math.inf, -math.inf, None, 0.0
            rx = power + dev.terr_gain_db - fade + self.cfg["terrestrial"]["fast_fading_std_db"] * dev.z_fast[k]
            snr = rx - self.noise_floor - self.anomalies.snr_penalty_db(t, channel)
            return snr, rx, None, 0.0
        if link == "satellite":
            st = self.satellite_state(t)
            if not st["visible"]:
                return -math.inf, -math.inf, None, 0.0
            sat = self.cfg["satellite"]
            el = st["elevation_deg"]
            fspl = 20 * math.log10(st["range_km"]) + 20 * math.log10(sat["frequency_hz"] / 1e6) + 32.45
            extra, _ = extra_loss_sample(el, dev.u_los[k], dev.z_sat[k])
            rx = power + sat["device_antenna_gain_dbi"] + sat["rx_gain_dbi"] - fspl - extra - fade
            return rx - self.sat_noise_floor, rx, el, abs(st["doppler_rate_hz_s"])
        return -math.inf, -math.inf, None, 0.0

    # ── control plane ─────────────────────────────────────────────────

    def issue_override(self, device_ids: list[int], fields: dict[str, Any], lease_s: float) -> int:
        """Queue a parameter override; it reaches each device at its next
        delivered uplink (class A) or by satellite broadcast during a pass."""
        for d in device_ids:
            self.pending[int(d)] = (dict(fields), float(lease_s), self.now)
        self.overrides_issued += len(device_ids)
        return len(device_ids)

    def _deliver_pending(self, dev: Device) -> None:
        item = self.pending.pop(dev.device_id, None)
        if item is None:
            return
        fields, lease, _ = item
        dev.override = fields
        dev.override_until = self.now + lease
        self.overrides_delivered += 1

    def set_base(self, dev: Device, params: TxParams) -> None:
        """Network-side base parameters (ADR / ML): call from on_uplink only."""
        dev.base = params

    # ── main loop ─────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        for dev in self.devices:
            for k, t in enumerate(dev.reading_times):
                self._push(float(t), _READING, "reading", (dev.device_id, k))
        t = self.tick_s
        while t <= self.duration:
            self._push(t, _TICK, "tick", None)
            t += self.tick_s
        while self._q:
            t, _, _, kind, payload = heapq.heappop(self._q)
            if t > self.duration:
                break
            self.now = t
            if kind == "reading":
                self._on_reading(*payload)
            elif kind == "tx_start":
                self._on_tx_start(payload)
            elif kind == "tx_end":
                self._on_tx_end(payload)
            elif kind == "tick":
                self._on_tick()
        return self.summary()

    def _on_reading(self, dev_id: int, k: int) -> None:
        dev = self.devices[dev_id]
        dev.stats["readings"] += 1
        if getattr(self.controller, "per_packet", False):
            params = self.controller.choose(self, dev, k)
        else:
            params = dev.effective_params(self.now)
        if params.link == "wait":
            dev.stats["withheld"] += 1
            return
        toa = self.toa[params.sf]
        t_tx = max(self.now, dev.next_tx_allowed)
        nxt = dev.reading_times[k + 1] if k + 1 < len(dev.reading_times) else self.duration
        if t_tx >= nxt or t_tx + toa > self.duration:
            dev.stats["dropped_duty"] += 1
            return
        chans = params.channels or tuple(self.channels)
        ch = chans[min(int(dev.u_channel[k] * len(chans)), len(chans) - 1)]
        self._pkt_id += 1
        dev.fcnt += 1
        pkt = Packet(self._pkt_id, dev_id, k, dev.fcnt, params.sf, int(params.tx_power_dbm),
                     float(ch), params.link, t_tx, toa)
        pkt.energy_j = tx_energy_joules(pkt.tx_power_dbm, toa)
        dev.next_tx_allowed = t_tx + toa * (1.0 + self.duty_off)
        if t_tx > self.now:
            self._push(t_tx, _TX_START, "tx_start", pkt)
        else:
            self._on_tx_start(pkt)

    def _on_tx_start(self, pkt: Packet) -> None:
        dev = self.devices[pkt.device_id]
        mid = pkt.start + 0.5 * pkt.toa
        snr, rx, el, rate = self.true_snr(dev, pkt.reading_idx, pkt.link, pkt.tx_power_dbm,
                                          pkt.channel, mid)
        pkt.snr_db, pkt.rx_power_dbm, pkt.elevation_deg, pkt.doppler_rate = snr, rx, el, rate
        if snr == -math.inf:
            pkt.no_gateway = True
        else:
            active = self._active[pkt.link]
            for o in active:
                if o.channel != pkt.channel or o.sf != pkt.sf or o.no_gateway:
                    continue
                if o.end <= pkt.start:
                    continue
                delta = pkt.rx_power_dbm - o.rx_power_dbm
                if delta >= self.capture_db:
                    o.collided = True
                elif delta <= -self.capture_db:
                    pkt.collided = True
                else:
                    o.collided = True
                    pkt.collided = True
            active.append(pkt)
        dev.stats["sent"] += 1
        dev.stats["energy_j"] += pkt.energy_j
        self._push(pkt.end, _TX_END, "tx_end", pkt)

    def _on_tx_end(self, pkt: Packet) -> None:
        if not pkt.no_gateway:
            act = self._active[pkt.link]
            act.remove(pkt)
            if not pkt.collided:
                # gateway must still be there at the end of the packet
                if pkt.link == "terrestrial" and self.anomalies.gateway_down(pkt.end):
                    pkt.no_gateway = True
                elif pkt.link == "satellite" and not self.satellite_state(pkt.end)["visible"]:
                    pkt.no_gateway = True
                else:
                    per = self.per_lookup(pkt.sf, pkt.snr_db, pkt.doppler_rate)
                    dev = self.devices[pkt.device_id]
                    pkt.per_failed = bool(dev.u_per[pkt.reading_idx] < per)
        self.packets.append(pkt)
        if pkt.delivered:
            dev = self.devices[pkt.device_id]
            dev.stats["delivered"] += 1
            rec = self.ns.receive(pkt)
            self.controller.on_uplink(self, dev, rec)
            self._deliver_pending(dev)          # class-A downlink after the uplink

    def _on_tick(self) -> None:
        now = self.now
        # satellite broadcast downlink
        if self.cfg["network"].get("satellite_broadcast_downlink") and self.pending:
            if self.satellite_state(now)["visible"]:
                for dev_id in list(self.pending):
                    self._deliver_pending(self.devices[dev_id])
        obs = self._observe(now)
        alerts = self.detector.observe(now, obs)
        self.tick_log.append({"t": now, "n_alerts": len(alerts)})
        self.controller.on_tick(self, now, alerts)

    BLOCK_S = 300.0

    def _observe(self, now: float) -> dict[str, Any]:
        """Network-server measurements for the detector (one tick)."""
        std = self.cfg["detector"]["noise_meas_std_db"]
        nf = {}
        if not self.anomalies.gateway_down(now):
            # the gateway measures RSSI on its channels (only while it is up)
            for ch in self.channels:
                lvl = self.noise_floor + self.anomalies.snr_penalty_db(now, ch) + std * self._noise_rng.standard_normal()
                nf[ch] = lvl
                self.ns.noise_floor[ch].append((now, lvl))
        # silence of the terrestrial gateway itself (uplinks received via the
        # satellite do not count as a sign of life of the terrestrial gateway)
        last_terr = self.ns.last_rx_by_gateway.get("terrestrial")
        gw_silent = (now - last_terr) if last_terr is not None else 0.0
        obs: dict[str, Any] = {"noise_floor_dbm": nf, "gw_silent_s": gw_silent}
        if now % self.BLOCK_S < 1e-6 or abs(now % self.BLOCK_S - self.BLOCK_S) < 1e-6:
            from policies.triggers import affected_devices   # NS-side helper
            win = self.ns.window(now, self.BLOCK_S)
            mean_snr = {}
            for ch in self.channels:
                v = [r.snr_db for r in win if r.channel == ch and r.link == "terrestrial"]
                if v:
                    mean_snr[ch] = (float(np.mean(v)), len(v))
            lost = sum(r.lost_before for r in win)
            terr = [d for d in self.devices if d.home_link == "terrestrial"]
            silent = [d.device_id for d in terr
                      if now > 2.5 * self.interval
                      and now - self.ns.last_seen.get(d.device_id, 0.0) > 2.5 * self.interval]
            obs["block"] = {
                "mean_snr": mean_snr,
                "loss": (lost, lost + len(win)),
                "uplinks": float(lost + len(win)),
                "silent": silent,
                "silent_fraction": len(silent) / max(1, len(terr)),
                "affected": affected_devices(self, now),
            }
        return obs

    # ── metrics ───────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        sent = len(self.packets)
        deliv = sum(p.delivered for p in self.packets)
        energy = sum(p.energy_j for p in self.packets)
        readings = sum(d.stats["readings"] for d in self.devices)
        out = {
            "sent": sent, "delivered": deliv,
            "pdr": deliv / sent if sent else float("nan"),
            "readings": readings,
            "reading_delivery_ratio": deliv / readings if readings else float("nan"),
            "energy_j": energy,
            "energy_per_delivered_mj": energy / deliv * 1e3 if deliv else float("nan"),
            "collided": sum(p.collided for p in self.packets),
            "no_gateway": sum(p.no_gateway for p in self.packets),
            "per_failed": sum(p.per_failed for p in self.packets),
            "withheld": sum(d.stats["withheld"] for d in self.devices),
            "dropped_duty": sum(d.stats["dropped_duty"] for d in self.devices),
            "overrides_issued": self.overrides_issued,
            "overrides_delivered": self.overrides_delivered,
        }
        for link in ("terrestrial", "satellite"):
            pk = [p for p in self.packets if p.link == link]
            d = sum(p.delivered for p in pk)
            out[f"sent_{link}"] = len(pk)
            out[f"pdr_{link}"] = d / len(pk) if pk else float("nan")
        sf_counts = defaultdict(int)
        for p in self.packets:
            sf_counts[p.sf] += 1
        for sf in range(7, 13):
            out[f"share_sf{sf}"] = sf_counts[sf] / sent if sent else 0.0
        if self.anomalies.events:
            out.update({f"det_{k}": v for k, v in
                        evaluate_detector(self.detector.alerts, self.anomalies).items()})
        return out

    def time_series(self, bin_s: float | None = None) -> dict[str, np.ndarray]:
        """Per-bin sent / delivered / energy (by transmission start time)."""
        bin_s = bin_s or self.cfg["metrics"]["pdr_bin_s"]
        nb = int(math.ceil(self.duration / bin_s))
        sent, deliv, energy = np.zeros(nb), np.zeros(nb), np.zeros(nb)
        for p in self.packets:
            i = min(int(p.start // bin_s), nb - 1)
            sent[i] += 1
            deliv[i] += p.delivered
            energy[i] += p.energy_j
        return {"t_start": np.arange(nb) * bin_s, "sent": sent, "delivered": deliv, "energy_j": energy}

    def window_metrics(self, t0: float, t1: float) -> dict[str, float]:
        pk = [p for p in self.packets if t0 <= p.start < t1]
        d = sum(p.delivered for p in pk)
        e = sum(p.energy_j for p in pk)
        return {"sent": len(pk), "delivered": d, "pdr": d / len(pk) if pk else float("nan"),
                "energy_j": e, "energy_per_delivered_mj": e / d * 1e3 if d else float("nan")}


# ═══════════════════════════════════════════════════════════════════════════
# Recovery time (metric definition in docs/PROTOCOL.md)
# ═══════════════════════════════════════════════════════════════════════════

def recovery_time(ts: dict[str, np.ndarray], t_event: float, cfg: dict[str, Any],
                  t_event_end: float | None = None) -> float:
    """Time from event start until PDR returns within 5 % of its pre-event level.

    * Pre-event PDR = PDR of the packets sent in [t_event − pre_window, t_event).
    * After t_event, PDR per bin of ``pdr_bin_s`` (10 min: one reporting
      period, so every device contributes about once per bin).
    * Judged until the horizon t_event_end + ``recovery_horizon_s``.
    * Recovery time = end of the last bin below (1 − tol)·PDR_pre, minus
      t_event; 0 if no bin is below; NaN (censored) if the last bin before
      the horizon is still below.
    """
    m = cfg["metrics"]
    bin_s = float(m["pdr_bin_s"])
    tol = float(m["recovery_tolerance"])
    t = ts["t_start"]
    pre = (t >= t_event - m["pre_event_window_s"]) & (t < t_event)
    s_pre, d_pre = ts["sent"][pre].sum(), ts["delivered"][pre].sum()
    if s_pre == 0:
        return float("nan")
    target = (1 - tol) * d_pre / s_pre
    end = (t_event_end if t_event_end is not None else t_event) + float(m["recovery_horizon_s"])
    idx = [i for i in range(len(t)) if t[i] >= t_event and t[i] + bin_s <= end]
    if not idx:
        return float("nan")
    below = [i for i in idx if ts["sent"][i] > 0 and ts["delivered"][i] / ts["sent"][i] < target]
    if not below:
        return 0.0
    if below[-1] == idx[-1]:
        return float("nan")
    return float(t[below[-1]] + bin_s - t_event)
