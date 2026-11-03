"""Machine learning classifier policy for LoRa parameter selection.

Uses scikit-learn's HistGradientBoostingClassifier trained on Oracle labels.
Features:
  - Mean SNR of last k packets
  - Last SNR
  - Estimated PER
  - Current SF and transmit power
  - Link type (terrestrial vs satellite)
  - Satellite elevation angle (deg)
  - Doppler rate (Hz/s)
  - Anomaly flag (0 or 1)
"""

from typing import Optional
import numpy as np

from policies.fixed import BasePolicy, TxParams
from policies.oracle import OraclePolicy


class MLClassifierPolicy(BasePolicy):
    """ML-based policy using HistGradientBoostingClassifier trained on Oracle labels."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__("MLClassifier")
        self.seed = seed
        self.model = None
        self._classes = None
        self._is_trained = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                random_state=self.seed,
            )
        except ImportError:
            self.model = None

    @staticmethod
    def extract_features(
        snr_history: list[float],
        per: float,
        current_sf: int,
        current_power: float,
        is_satellite: bool = False,
        satellite_elevation: float = 0.0,
        doppler_rate: float = 0.0,
        anomaly_flag: int = 0,
        k: int = 5,
    ) -> np.ndarray:
        """Extract fixed-length feature vector."""
        if len(snr_history) >= 1:
            recent_snr = snr_history[-k:]
            mean_snr = float(np.mean(recent_snr))
            last_snr = float(recent_snr[-1])
        else:
            mean_snr = -10.0
            last_snr = -10.0

        return np.array([
            mean_snr,
            last_snr,
            float(per),
            float(current_sf),
            float(current_power),
            1.0 if is_satellite else 0.0,
            float(satellite_elevation),
            float(doppler_rate),
            float(anomaly_flag),
        ], dtype=np.float32)

    def train_on_oracle(self, n_samples: int = 1000, seed: int = 42) -> None:
        """Generate synthetic diverse channel scenarios, label with Oracle, and train model."""
        if self.model is None:
            return

        rng = np.random.default_rng(seed)
        oracle = OraclePolicy()

        X = []
        y = []

        # Generate synthetic scenarios
        for _ in range(n_samples):
            ref_snr = float(rng.uniform(-25.0, 15.0))
            is_sat = bool(rng.random() < 0.3)
            elev = float(rng.uniform(10.0, 90.0)) if is_sat else 0.0
            d_rate = float(rng.uniform(-200.0, 200.0)) if is_sat else 0.0
            anomaly = int(rng.random() < 0.2)
            cur_sf = int(rng.choice([7, 8, 9, 10, 11, 12]))
            cur_pwr = int(rng.choice([2, 5, 8, 11, 14]))
            per = float(rng.uniform(0.0, 0.5))

            history = [ref_snr + float(rng.normal(0, 1)) for _ in range(5)]
            feat = self.extract_features(
                history, per, cur_sf, cur_pwr, is_sat, elev, d_rate, anomaly
            )

            # Oracle choice
            link_name = "satellite" if is_sat else "terrestrial"
            oracle_tx = oracle.decide_optimal_params(
                device_id=0,
                ref_snr_db=ref_snr,
                ref_power_dbm=14.0,
                link=link_name,
                doppler_rate_hz_s=d_rate,
            )

            # Label encoded as (SF - 7) * 5 + power_idx
            power_idx = {2: 0, 5: 1, 8: 2, 11: 3, 14: 4}.get(oracle_tx.tx_power_dbm, 4)
            label = (oracle_tx.sf - 7) * 5 + power_idx

            X.append(feat)
            y.append(label)

        X_arr = np.array(X)
        y_arr = np.array(y)

        self.model.fit(X_arr, y_arr)
        self._classes = np.unique(y_arr)
        self._is_trained = True

    def get_tx_params(self, device_id: int, sim_time: float) -> TxParams:
        """Default fallback."""
        return TxParams(sf=7, tx_power_dbm=14, channel=868.1, link="terrestrial")

    def predict_tx_params(
        self,
        features: np.ndarray,
        channel: float = 868.1,
        link: str = "terrestrial",
    ) -> TxParams:
        """Predict transmission parameters given an extracted feature vector."""
        if not self._is_trained or self.model is None:
            # Fallback if scikit-learn is not available or not trained
            return TxParams(sf=12, tx_power_dbm=14, channel=channel, link=link)

        pred_label = int(self.model.predict(features.reshape(1, -1))[0])
        sf = 7 + (pred_label // 5)
        sf = max(7, min(12, sf))
        pwr_map = [2, 5, 8, 11, 14]
        pwr = pwr_map[pred_label % 5]

        return TxParams(sf=sf, tx_power_dbm=pwr, channel=channel, link=link)
