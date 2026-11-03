"""Device abstractions and population generation for LoRa network simulation.

Provides:
  - Device dataclass representing an IoT sensor node
  - Helper to generate spatial distributions of devices
"""

from dataclasses import dataclass, field
import math
import numpy as np

from policies.fixed import TxParams


@dataclass
class Device:
    """Represents a LoRa IoT sensor device in the network."""
    device_id: int
    x_km: float = 0.0
    y_km: float = 0.0
    distance_to_gw_km: float = 1.0
    tx_params: TxParams = field(default_factory=lambda: TxParams(
        sf=12, tx_power_dbm=14, channel=868.1, link="terrestrial"
    ))
    snr_history: list[float] = field(default_factory=list)
    packets_sent: int = 0
    packets_delivered: int = 0
    total_energy_mj: float = 0.0

    @property
    def pdr(self) -> float:
        """Packet Delivery Ratio for this device."""
        if self.packets_sent == 0:
            return 1.0
        return self.packets_delivered / self.packets_sent

    def record_tx(self, delivered: bool, snr_db: float, energy_mj: float) -> None:
        """Update device statistics after a transmission attempt."""
        self.packets_sent += 1
        if delivered:
            self.packets_delivered += 1
            self.snr_history.append(snr_db)
        self.total_energy_mj += energy_mj


def create_uniform_devices(
    n_devices: int,
    radius_km: float = 5.0,
    min_radius_km: float = 0.1,
    seed: int = 42,
) -> list[Device]:
    """Generate devices uniformly distributed in a 2D disk around gateway (0, 0).

    Parameters
    ----------
    n_devices : int
        Number of devices to instantiate.
    radius_km : float
        Maximum cell radius in km.
    min_radius_km : float
        Minimum distance from gateway to avoid singularity (default 0.1 km = 100 m).
    seed : int
        Random seed for device positioning.

    Returns
    -------
    list[Device]
        List of configured devices.
    """
    rng = np.random.default_rng(seed)
    devices = []

    # Uniform distribution over a disk: r = R * sqrt(u), theta = 2*pi*v
    u = rng.uniform((min_radius_km / radius_km) ** 2, 1.0, size=n_devices)
    theta = rng.uniform(0, 2 * math.pi, size=n_devices)
    radii = radius_km * np.sqrt(u)

    for i in range(n_devices):
        r = float(radii[i])
        th = float(theta[i])
        x = r * math.cos(th)
        y = r * math.sin(th)
        devices.append(
            Device(
                device_id=i,
                x_km=x,
                y_km=y,
                distance_to_gw_km=r,
                tx_params=TxParams(
                    sf=12,
                    tx_power_dbm=14,
                    channel=868.1,
                    link="terrestrial",
                ),
            )
        )

    return devices
