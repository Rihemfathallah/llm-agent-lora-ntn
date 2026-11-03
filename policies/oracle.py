"""Oracle baseline policy for LoRa parameter selection.

The Oracle serves as a theoretical upper bound (prescient benchmark).
For each decision, it evaluates every (SF, transmit_power) combination against
the true channel condition (true SNR and Doppler rate) and selects the
lowest-energy configuration that achieves PER <= 1%.
"""

from typing import Optional
from policies.fixed import BasePolicy, TxParams
from phy.css import time_on_air
from netsim.simulator import tx_energy_joules

# Standard SNR sensitivity thresholds for AWGN channel at PER ~ 1%
SNR_REQ = {
    7: -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0,
}


class OraclePolicy(BasePolicy):
    """Oracle controller that selects the optimal (SF, power) configuration."""

    def __init__(self, target_per: float = 0.01, payload_bytes: int = 20) -> None:
        super().__init__("Oracle")
        self.target_per = target_per
        self.payload_bytes = payload_bytes

    def get_tx_params(self, device_id: int, sim_time: float) -> TxParams:
        """Default fallback if no channel info is provided."""
        return TxParams(sf=7, tx_power_dbm=14, channel=868.1, link="terrestrial")

    def decide_optimal_params(
        self,
        device_id: int,
        ref_snr_db: float,
        ref_power_dbm: float = 14.0,
        channel: float = 868.1,
        link: str = "terrestrial",
        doppler_rate_hz_s: float = 0.0,
    ) -> TxParams:
        """Evaluate all candidate settings and select minimum energy with PER <= target_per.

        Parameters
        ----------
        device_id : int
            Target device identifier.
        ref_snr_db : float
            True channel SNR measured/expected at ref_power_dbm.
        ref_power_dbm : float
            Transmit power at which ref_snr_db was evaluated.
        channel : float
            Operating frequency channel in MHz.
        link : str
            Link type ('terrestrial' or 'satellite').
        doppler_rate_hz_s : float
            Doppler rate of change (Hz/s).

        Returns
        -------
        TxParams
            Optimal transmission parameters.
        """
        # Noise-equivalent SNR scaling: SNR(P) = ref_snr_db + (P - ref_power_dbm)
        best_params: Optional[TxParams] = None
        min_energy = float("inf")
        min_per_seen = float("inf")
        fallback_params: Optional[TxParams] = None

        candidate_sfs = [7, 8, 9, 10, 11, 12]
        candidate_powers = [2, 5, 8, 11, 14]

        for sf in candidate_sfs:
            toa = time_on_air(sf, self.payload_bytes)

            # Check Doppler impairment for satellite links
            # (SF11 and SF12 fail if Doppler rate > ~100 Hz/s)
            if link == "satellite" and abs(doppler_rate_hz_s) > 100.0 and sf >= 11:
                continue

            req_snr = SNR_REQ[sf]

            for pwr in candidate_powers:
                snr_at_pwr = ref_snr_db + (pwr - ref_power_dbm)
                energy = tx_energy_joules(pwr, toa) * 1e3

                # Link margin relative to requirement
                margin = snr_at_pwr - req_snr

                # Approximate PER check: link closes with PER <= 1% if margin >= 0
                closes = (margin >= 0.0)

                if closes:
                    if energy < min_energy:
                        min_energy = energy
                        best_params = TxParams(
                            sf=sf,
                            tx_power_dbm=pwr,
                            channel=channel,
                            link=link,
                        )
                else:
                    # Track fallback with closest margin
                    loss = -margin
                    if loss < min_per_seen:
                        min_per_seen = loss
                        fallback_params = TxParams(
                            sf=sf,
                            tx_power_dbm=14,
                            channel=channel,
                            link=link,
                        )

        if best_params is not None:
            return best_params

        # If link cannot close at PER <= target_per, maximize delivery probability
        return fallback_params or TxParams(
            sf=12, tx_power_dbm=14, channel=channel, link=link
        )
