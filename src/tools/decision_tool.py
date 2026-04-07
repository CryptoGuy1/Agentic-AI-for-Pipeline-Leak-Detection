import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# Dueling DQN (matches training notebook architecture)
# =========================================================
class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int = 22, output_dim: int = 5, dropout: float = 0.15):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.value_stream[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.adv_stream[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC-dropout inference.
        Keep n_samples low for deployment/debugging speed.
        """
        self.train()
        qs = torch.stack([self(x) for _ in range(n_samples)], dim=0)
        self.eval()

        mean_q = qs.mean(0)
        std_q = qs.std(0)

        action = mean_q.argmax(1)

        top2 = mean_q.topk(2, dim=1).values
        gap = top2[:, 0] - top2[:, 1]

        conf = (gap / (std_q.mean(1) + 1e-6)).clamp(0, 10) / 10.0
        return mean_q, std_q, action, conf


# =========================================================
# Decision Tool for live deployment
# =========================================================
class DecisionTool:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        mc_dropout_samples: int = 5,
        window_size: int = 20,
    ):
        self.device = torch.device(device)
        self.model_path = model_path
        self.mc_dropout_samples = mc_dropout_samples
        self.window_size = window_size

        ckpt = torch.load(model_path, map_location=self.device)

        self.input_dim = int(ckpt.get("input_dim", 22))
        self.n_actions = int(ckpt.get("n_actions", 5))

        self.sensor_cols = ckpt.get(
            "sensor_cols",
            ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]
        )
        self.all_feat_cols = ckpt.get(
            "all_feat_cols",
            self.sensor_cols +
            [f"d{c}" for c in self.sensor_cols] +
            [f"s{c}" for c in self.sensor_cols]
        )

        self.feat_scaler_mean = np.array(ckpt["feat_scaler_mean"], dtype=np.float32)
        self.feat_scaler_scale = np.array(ckpt["feat_scaler_scale"], dtype=np.float32)

        self.anom_p1 = float(ckpt["anom_p1"])
        self.anom_p99 = float(ckpt["anom_p99"])

        self.model = DuelingDQN(
            input_dim=self.input_dim,
            output_dim=self.n_actions,
            dropout=0.15,
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.actions = {
            0: "monitor",
            1: "increase_sampling",
            2: "request_verification",
            3: "raise_alert",
            4: "emergency_shutdown",
        }

    def normalize_anomaly(self, anomaly_score: float) -> float:
        x = (anomaly_score - self.anom_p1) / (self.anom_p99 - self.anom_p1 + 1e-8)
        return float(np.clip(x, 0.0, 1.0))

    def scale_sensor_features(self, feat_vec_21: np.ndarray) -> np.ndarray:
        feat_vec_21 = np.asarray(feat_vec_21, dtype=np.float32)
        if feat_vec_21.shape[0] != len(self.all_feat_cols):
            raise ValueError(
                f"Expected {len(self.all_feat_cols)} scaled features, got {feat_vec_21.shape[0]}"
            )
        return (feat_vec_21 - self.feat_scaler_mean) / (self.feat_scaler_scale + 1e-8)

    def build_state_from_window(
        self,
        sensor_window: np.ndarray,
        anomaly_score_raw: float,
    ) -> np.ndarray:
        """
        sensor_window shape: (20, 7)
        state = [anomaly_norm] + current(7) + delta(7) + std(7)
        """
        sensor_window = np.asarray(sensor_window, dtype=np.float32)

        expected_shape = (self.window_size, len(self.sensor_cols))
        if sensor_window.shape != expected_shape:
            raise ValueError(
                f"sensor_window must have shape {expected_shape}, got {sensor_window.shape}"
            )

        current = sensor_window[-1, :]
        delta = sensor_window[-1, :] - sensor_window[0, :]
        std = sensor_window.std(axis=0)

        feat_21 = np.concatenate([current, delta, std], axis=0)
        feat_21_scaled = self.scale_sensor_features(feat_21)

        anomaly_norm = self.normalize_anomaly(anomaly_score_raw)

        state = np.concatenate(
            [np.array([anomaly_norm], dtype=np.float32), feat_21_scaled],
            axis=0
        ).astype(np.float32)

        if state.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected state dim {self.input_dim}, got {state.shape[0]}"
            )

        return state

    def decide(
        self,
        state: np.ndarray,
        use_mc_dropout: bool = False,
    ) -> Tuple[int, np.ndarray, np.ndarray, float]:
        """
        Returns:
            action_int, q_values, q_std, policy_conf
        """
        state = np.asarray(state, dtype=np.float32)

        if state.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected state dim {self.input_dim}, got {state.shape[0]}"
            )

        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if use_mc_dropout:
                mean_q, std_q, action, conf = self.model.predict_with_uncertainty(
                    x, n_samples=self.mc_dropout_samples
                )
                action_int = int(action.item())
                q_values = mean_q.squeeze(0).cpu().numpy()
                q_std = std_q.squeeze(0).cpu().numpy()
                policy_conf = float(conf.item())
            else:
                q = self.model(x)
                q_values = q.squeeze(0).cpu().numpy()
                q_std = np.zeros_like(q_values, dtype=np.float32)
                action_int = int(np.argmax(q_values))

                # simple deterministic confidence proxy
                sorted_q = np.sort(q_values)[::-1]
                gap = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) > 1 else 0.0
                policy_conf = float(np.clip(gap / 5.0, 0.0, 1.0))

        return action_int, q_values, q_std, policy_conf