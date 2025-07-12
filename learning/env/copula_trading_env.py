import gym
import numpy as np
import pandas as pd
from learning.copula.t_factor_copula import TFactorCopula

class CopulaTradingEnv(gym.Env):
    def __init__(self, data_source="real", window_size=30, initial_cash=1e6):
        super(CopulaTradingEnv, self).__init__()

        # === Load data ===
        path = f"data/mid_dimension/real_asset_log_returns_{data_source}.csv"
        self.df = pd.read_csv(path, index_col=0)
        self.data = self.df.values
        self.asset_dim = self.data.shape[1]
        self.window_size = window_size
        self.initial_cash = initial_cash

        # === Fit t-Factor Copula with fixed latent_dim=3 ===
        self.copula = TFactorCopula(latent_dim=3)
        print(f"🔄 Fitting t-Factor Copula | fixed latent_dim=3")
        self.copula.fit(self.data)

        # === Define observation & action spaces ===
        obs_dim = self.window_size * (self.asset_dim + self.copula.latent_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.asset_dim,), dtype=np.float32
        )

        # === Internal state ===
        self.current_step = None
        self.cash = None
        self.portfolio_value = None
        self.done = False

        print(f"✅ CopulaTradingEnv initialized | obs_dim={obs_dim}, latent_dim=3, asset_dim={self.asset_dim}")

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.portfolio_value = self.cash
        self.done = False
        obs = self._compute_observation()
        print(f"📏 reset() observation shape: {obs.shape}")
        return obs

    def step(self, action):
        """Take one step in the environment."""
        if self.done:
            raise RuntimeError("Step called after environment is done. Please reset.")

        # === Normalize action weights and ensure float type ===
        weights = np.clip(action, 0, 1).astype(np.float64)  # 👈 修复整数导致的类型问题
        weights /= np.sum(weights) + 1e-8

        # === Get returns and compute reward ===
        returns = self.data[self.current_step]
        reward = self.portfolio_value * np.dot(weights, returns)
        self.portfolio_value += reward

        # === Advance environment state ===
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        obs = self._compute_observation()
        info = {"portfolio_value": self.portfolio_value}

        return obs, reward, self.done, info

    def _compute_observation(self):
        """Compute observation: raw returns + copula latent factors."""
        start_idx = max(0, self.current_step - self.window_size)
        raw_obs = self.data[start_idx:self.current_step].flatten()
        latent_obs = self.copula.transform(self.data[start_idx:self.current_step]).flatten()
        combined_obs = np.concatenate([raw_obs, latent_obs])
        return combined_obs.astype(np.float32)









