"""Event-driven LoRa network simulator — pure ALOHA with duty-cycle and capture.

Implements Phase 3 of the project specification (Section 3.5):
  - heapq-based discrete-event simulation
  - Pure ALOHA: collision if two packets overlap on the same channel AND
    same SF, unless the stronger one is ≥ 6 dB above the other (capture)
  - Different SFs are treated as orthogonal (stated simplification)
  - 1 % duty cycle: after transmitting for ToA seconds the device is
    silent for 99 · ToA seconds
  - Packet success = survived collisions AND Bernoulli draw on PER table
  - Energy model: E = V · I_tx(P_tx) · ToA  (SX1276 datasheet currents)

Sanity check target: pure ALOHA throughput S = G · e^{-2G} peaks at
≈ 18.4 % when G = 0.5.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np

from phy.css import time_on_air, SNR_THRESHOLDS_DB

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Constants
# ═══════════════════════════════════════════════════════════════════════════

# EU868 default channels (MHz)
EU868_CHANNELS = [868.1, 868.3, 868.5]

# Duty-cycle multiplier: device silent for DUTY_CYCLE_OFF_RATIO · ToA
# after each transmission (1 % duty cycle → 99 × ToA off-time).
DUTY_CYCLE_OFF_RATIO = 99

# Capture-effect threshold (dB).  A packet survives a collision only if
# its received power is at least CAPTURE_THRESHOLD_DB above every
# overlapping co-channel / co-SF packet.
CAPTURE_THRESHOLD_DB = 6.0

# SX1276 supply current (mA) vs output power (dBm).
# Extracted from Semtech SX1276 datasheet Table 10 (PA_BOOST).
SX1276_CURRENT_MA: dict[int, float] = {
    2:  29.0,
    3:  31.0,
    4:  32.0,
    5:  34.0,
    6:  35.0,
    7:  37.0,
    8:  40.0,
    9:  43.0,
   10:  46.0,
   11:  52.0,
   12:  58.0,
   13:  68.0,
   14:  82.0,
}

# Supply voltage (V) — typical for a 3.3 V LoRa module
SUPPLY_VOLTAGE = 3.3


def tx_current_ma(p_tx_dbm: int) -> float:
    """Return SX1276 transmit current (mA) for a given output power.

    Linearly interpolates between datasheet entries if needed.
    """
    if p_tx_dbm in SX1276_CURRENT_MA:
        return SX1276_CURRENT_MA[p_tx_dbm]
    powers = sorted(SX1276_CURRENT_MA.keys())
    p_clamped = max(powers[0], min(powers[-1], p_tx_dbm))
    return float(np.interp(p_clamped,
                           powers,
                           [SX1276_CURRENT_MA[p] for p in powers]))


def tx_energy_joules(p_tx_dbm: int, toa_s: float) -> float:
    """Transmit energy  E = V · I_tx(P_tx) · ToA   (joules)."""
    return SUPPLY_VOLTAGE * tx_current_ma(p_tx_dbm) * 1e-3 * toa_s


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Event types and Packet record
# ═══════════════════════════════════════════════════════════════════════════

class EventType(Enum):
    TX_START = auto()         # a device begins transmitting
    TX_END = auto()           # transmission ends → resolve collisions
    SCHEDULE_TX = auto()      # wake-up to schedule a new transmission
    CUSTOM = auto()           # hook for policies / anomaly injection


@dataclass(order=True)
class Event:
    """A single event in the discrete-event queue.

    Ordered by *time* (ties broken by a monotonic counter).
    """
    time: float
    _seq: int = field(compare=True, repr=False)
    type: EventType = field(compare=False)
    data: dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class Packet:
    """Record of a single transmission attempt."""
    pkt_id: int
    device_id: int
    sf: int
    tx_power_dbm: int
    channel: float               # MHz
    link: str                    # "terrestrial" or "satellite"
    start_time: float            # simulation seconds
    toa: float                   # seconds
    end_time: float              # start_time + toa
    rx_power_dbm: float          # received power at the gateway (dBm)
    snr_db: float                # received SNR (dB)
    doppler_rate: float          # Hz/s (0 for terrestrial)
    energy_j: float              # transmit energy (J)
    collided: bool = False
    per_failed: bool = False

    @property
    def delivered(self) -> bool:
        return not self.collided and not self.per_failed


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Core simulator
# ═══════════════════════════════════════════════════════════════════════════

class Simulator:
    """Discrete-event LoRa network simulator.

    Parameters
    ----------
    duration_s : float
        Total simulated time (seconds).
    channels : list[float]
        Available channels in MHz.
    payload_bytes : int
        Application payload size (bytes).
    seed : int
        Random seed for reproducibility.
    per_lookup : callable or None
        ``per_lookup(sf, snr_db, doppler_rate) -> float``
        If *None*, use a simple threshold model (PER = 0 above threshold,
        PER = 1 below).
    """

    def __init__(
        self,
        duration_s: float = 3600.0,
        channels: list[float] | None = None,
        payload_bytes: int = 20,
        seed: int = 42,
        per_lookup: Callable[[int, float, float], float] | None = None,
    ) -> None:
        self.duration = duration_s
        self.channels = channels or EU868_CHANNELS
        self.payload_bytes = payload_bytes
        self.rng = np.random.default_rng(seed)
        self.per_lookup = per_lookup

        # Event machinery
        self._seq = 0                      # monotonic tie-breaker
        self._queue: list[Event] = []      # min-heap
        self.now: float = 0.0              # current simulation time

        # Tracking
        self.packets: list[Packet] = []
        self._pkt_counter = 0
        self._active_tx: list[Packet] = [] # packets currently on air

        # Per-device state
        self._duty_off_until: dict[int, dict[float, float]] = {}
        # _duty_off_until[device_id][channel] = earliest next TX time

        # Optional hooks
        self._event_hooks: dict[EventType, list[Callable]] = {
            et: [] for et in EventType
        }

    # ── event helpers ──────────────────────────────────────────────────

    def schedule_event(
        self, time: float, etype: EventType, data: dict | None = None
    ) -> None:
        """Insert an event into the queue."""
        self._seq += 1
        heapq.heappush(
            self._queue, Event(time, self._seq, etype, data or {})
        )

    def register_hook(self, etype: EventType, fn: Callable) -> None:
        """Register a callback that fires after each event of *etype*."""
        self._event_hooks[etype].append(fn)

    # ── duty-cycle enforcement ────────────────────────────────────────

    def _duty_available(self, device_id: int, channel: float) -> float:
        """Earliest time *device_id* may transmit on *channel*."""
        return self._duty_off_until.get(device_id, {}).get(channel, 0.0)

    def _duty_mark(
        self, device_id: int, channel: float, toa: float, tx_start: float
    ) -> None:
        """After a TX of *toa* seconds, mark the duty-cycle silence window.

        1 % duty cycle  ⟹  silent for 99 · ToA after the transmission ends.
        """
        off_until = tx_start + toa + DUTY_CYCLE_OFF_RATIO * toa
        self._duty_off_until.setdefault(device_id, {})[channel] = off_until

    # ── collision resolution (pure ALOHA + capture effect) ────────────

    def _resolve_collisions(self, pkt: Packet) -> None:
        """Check *pkt* against all currently active transmissions.

        Pure ALOHA rule (Section 3.5 Phase 3):
          - Collision if another packet is on the **same channel** AND
            **same SF** and **overlaps in time**.
          - Different SFs are treated as orthogonal (simplification).
          - Capture effect: the stronger packet survives if it is at
            least 6 dB above every interferer.
        """
        for other in self._active_tx:
            if other.pkt_id == pkt.pkt_id:
                continue
            if other.channel != pkt.channel:
                continue
            if other.sf != pkt.sf:
                continue  # orthogonal SFs

            # Time overlap check
            if pkt.start_time >= other.end_time:
                continue
            if pkt.end_time <= other.start_time:
                continue

            # Overlapping on same channel and same SF → potential collision.
            delta = pkt.rx_power_dbm - other.rx_power_dbm

            if delta >= CAPTURE_THRESHOLD_DB:
                # pkt captures — the weaker packet (other) is destroyed
                other.collided = True
            elif delta <= -CAPTURE_THRESHOLD_DB:
                # other captures — pkt is destroyed
                pkt.collided = True
            else:
                # Neither is strong enough to capture → both destroyed
                pkt.collided = True
                other.collided = True

    # ── PER evaluation ────────────────────────────────────────────────

    def _per_check(self, pkt: Packet) -> None:
        """Bernoulli draw using the PER lookup table (or threshold model)."""
        if pkt.collided:
            return  # already failed

        if self.per_lookup is not None:
            per = self.per_lookup(pkt.sf, pkt.snr_db, pkt.doppler_rate)
        else:
            # Simple threshold model: PER = 0 if above threshold, else 1
            threshold = SNR_THRESHOLDS_DB[pkt.sf]
            per = 0.0 if pkt.snr_db >= threshold else 1.0

        if self.rng.random() < per:
            pkt.per_failed = True

    # ── transmit a packet ─────────────────────────────────────────────

    def transmit(
        self,
        device_id: int,
        sf: int,
        tx_power_dbm: int,
        channel: float,
        link: str,
        rx_power_dbm: float,
        snr_db: float,
        doppler_rate: float = 0.0,
    ) -> int | None:
        """Schedule a packet transmission at the current simulation time.

        Respects the duty-cycle constraint: if the device is still in its
        off-period on the requested channel, the transmission is **deferred**
        to the earliest available time.  If that deferred time falls beyond
        the simulation horizon, the packet is dropped (returns None).

        Returns
        -------
        int or None
            Packet ID, or None if the duty-cycle constraint pushes the
            transmission beyond the simulation duration.
        """
        toa = time_on_air(self.payload_bytes, sf)
        earliest = max(self.now, self._duty_available(device_id, channel))

        if earliest + toa > self.duration:
            return None  # would exceed simulation window

        self._pkt_counter += 1
        pkt_id = self._pkt_counter

        pkt = Packet(
            pkt_id=pkt_id,
            device_id=device_id,
            sf=sf,
            tx_power_dbm=tx_power_dbm,
            channel=channel,
            link=link,
            start_time=earliest,
            toa=toa,
            end_time=earliest + toa,
            rx_power_dbm=rx_power_dbm,
            snr_db=snr_db,
            doppler_rate=doppler_rate,
            energy_j=tx_energy_joules(tx_power_dbm, toa),
        )

        # Mark duty-cycle off-period
        self._duty_mark(device_id, channel, toa, earliest)

        # Schedule TX_START and TX_END events
        self.schedule_event(earliest, EventType.TX_START, {"packet": pkt})
        self.schedule_event(pkt.end_time, EventType.TX_END, {"packet": pkt})

        return pkt_id

    # ── main loop ─────────────────────────────────────────────────────

    def run(self) -> list[Packet]:
        """Execute the simulation until the queue is empty or time expires.

        Returns
        -------
        list[Packet]
            All transmission records (including failed ones).
        """
        while self._queue:
            event = heapq.heappop(self._queue)
            if event.time > self.duration:
                break
            self.now = event.time

            if event.type == EventType.TX_START:
                pkt: Packet = event.data["packet"]
                # Check for collisions with packets currently on air
                self._resolve_collisions(pkt)
                self._active_tx.append(pkt)

            elif event.type == EventType.TX_END:
                pkt = event.data["packet"]
                # Remove from active transmissions
                self._active_tx = [
                    p for p in self._active_tx if p.pkt_id != pkt.pkt_id
                ]
                # PER draw (if not already collided)
                self._per_check(pkt)
                self.packets.append(pkt)

            elif event.type == EventType.SCHEDULE_TX:
                pass  # handled by hooks

            # Fire registered hooks
            for hook in self._event_hooks[event.type]:
                hook(self, event)

        return self.packets

    # ── metrics ───────────────────────────────────────────────────────

    def metrics(self) -> dict[str, float]:
        """Compute aggregate performance metrics.

        Returns dict with:
          - total_packets
          - delivered_packets
          - collided_packets
          - per_failed_packets
          - pdr              (packet delivery ratio)
          - total_energy_j
          - energy_per_delivered_mj  (mJ per delivered packet)
        """
        total = len(self.packets)
        if total == 0:
            return {
                "total_packets": 0,
                "delivered_packets": 0,
                "collided_packets": 0,
                "per_failed_packets": 0,
                "pdr": 0.0,
                "total_energy_j": 0.0,
                "energy_per_delivered_mj": float("inf"),
            }
        delivered = sum(1 for p in self.packets if p.delivered)
        collided = sum(1 for p in self.packets if p.collided)
        per_failed = sum(1 for p in self.packets if p.per_failed)
        total_energy = sum(p.energy_j for p in self.packets)

        return {
            "total_packets": total,
            "delivered_packets": delivered,
            "collided_packets": collided,
            "per_failed_packets": per_failed,
            "pdr": delivered / total,
            "total_energy_j": total_energy,
            "energy_per_delivered_mj": (
                total_energy / delivered * 1e3 if delivered > 0
                else float("inf")
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Traffic generator helpers
# ═══════════════════════════════════════════════════════════════════════════

def generate_poisson_traffic(
    sim: Simulator,
    n_devices: int,
    mean_interval_s: float = 600.0,
    sf: int = 7,
    tx_power_dbm: int = 14,
    channel: float | None = None,
    link: str = "terrestrial",
    snr_db: float = 0.0,
    rx_power_dbm: float = -100.0,
    doppler_rate: float = 0.0,
) -> None:
    """Pre-schedule Poisson traffic for *n_devices* over the simulation.

    Each device sends a 20-byte payload on average every *mean_interval_s*
    seconds with exponentially distributed inter-arrivals (random jitter as
    specified in Section 3.2).

    If *channel* is None, devices are assigned to channels round-robin.
    """
    for dev_id in range(n_devices):
        ch = channel if channel is not None else sim.channels[dev_id % len(sim.channels)]
        t = sim.rng.exponential(mean_interval_s)
        while t < sim.duration:
            sim.schedule_event(t, EventType.SCHEDULE_TX, {
                "device_id": dev_id,
                "sf": sf,
                "tx_power_dbm": tx_power_dbm,
                "channel": ch,
                "link": link,
                "rx_power_dbm": rx_power_dbm,
                "snr_db": snr_db,
                "doppler_rate": doppler_rate,
            })
            t += sim.rng.exponential(mean_interval_s)


def _default_schedule_hook(sim: Simulator, event: Event) -> None:
    """Default hook for SCHEDULE_TX events: call ``sim.transmit``."""
    if event.type != EventType.SCHEDULE_TX:
        return
    d = event.data
    sim.transmit(
        device_id=d["device_id"],
        sf=d["sf"],
        tx_power_dbm=d["tx_power_dbm"],
        channel=d["channel"],
        link=d["link"],
        rx_power_dbm=d["rx_power_dbm"],
        snr_db=d["snr_db"],
        doppler_rate=d.get("doppler_rate", 0.0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Pure-ALOHA throughput sanity check
# ═══════════════════════════════════════════════════════════════════════════

def aloha_sanity_check(
    n_devices_list: list[int] | None = None,
    duration_s: float = 3600.0,
    mean_interval_s: float = 600.0,
    sf: int = 7,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Run the simulator at varying loads and compare to ALOHA theory.

    Theory:  S = G · e^{-2G}  with  G = n_devices · ToA / (mean_interval · n_channels)
    Peak throughput ≈ 18.4 % at G = 0.5.

    Returns a list of dicts with keys: n_devices, G, S_theory, S_sim, pdr.
    """
    from phy.css import time_on_air as _toa

    if n_devices_list is None:
        n_devices_list = [10, 50, 100, 200, 500, 1000, 2000, 5000]

    toa = _toa(20, sf)
    n_channels = len(EU868_CHANNELS)
    results = []

    for n_dev in n_devices_list:
        # Offered load per channel
        G = n_dev * toa / (mean_interval_s * n_channels)
        S_theory = G * math.exp(-2 * G)

        sim = Simulator(
            duration_s=duration_s,
            payload_bytes=20,
            seed=seed,
            per_lookup=lambda _sf, _snr, _dr: 0.0,  # no PER — isolate collisions
        )
        sim.register_hook(EventType.SCHEDULE_TX, _default_schedule_hook)

        # All devices share the same high SNR so PER = 0 and no capture issues.
        # Give each device the same rx_power so collisions are symmetric.
        generate_poisson_traffic(
            sim,
            n_devices=n_dev,
            mean_interval_s=mean_interval_s,
            sf=sf,
            tx_power_dbm=14,
            link="terrestrial",
            snr_db=20.0,
            rx_power_dbm=-80.0,
            doppler_rate=0.0,
        )

        sim.run()
        m = sim.metrics()
        S_sim = m["delivered_packets"] * toa / (duration_s * n_channels)

        results.append({
            "n_devices": n_dev,
            "G": G,
            "S_theory": S_theory,
            "S_sim": S_sim,
            "pdr": m["pdr"],
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 6.  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Pure ALOHA sanity check (SF7, 125 kHz, 20 B, 3 channels)")
    print("=" * 65)
    print(f"  Duty cycle: 1 %  (off for {DUTY_CYCLE_OFF_RATIO} x ToA)")
    print(f"  Capture threshold: {CAPTURE_THRESHOLD_DB} dB")
    print()

    results = aloha_sanity_check(
        n_devices_list=[10, 50, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000],
    )

    header = f"{'N_dev':>6} {'G':>8} {'S_theory':>10} {'S_sim':>10} {'PDR':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['n_devices']:6d} "
            f"{r['G']:8.4f} "
            f"{r['S_theory']:10.4f} "
            f"{r['S_sim']:10.4f} "
            f"{r['pdr']:8.3f}"
        )

    print()
    print("Theory peak: S ~ 0.184 at G = 0.5")
