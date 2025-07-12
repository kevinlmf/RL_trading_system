import os
import gym
import numpy as np
import pandas as pd

class BaselineTradingEnv(gym.Env):
    """
    A multi-asset trading environment using raw historical features.
    """
    def __init__(self,
                 data_source="real",
                 data_path_dict=None,
                 window_size=30,
                 initial_cash=1e6):
        super(BaselineTradingEnv, self).__init__()

        # === 1. Data source ===
        default_paths = {
            "simulated": "data/mid_dimension/simulated_copula_returns.csv",
            "real": "data/mid_dimension/real_asset_log_returns.csv",
            "extreme": "data/mid_dimension/real_asset_log_returns_extreme.csv",
        }
        if data_path_dict is None:
            data_path_dict = default_paths

        # Fallback: auto-generate extreme data if missing
        if data_source == "extreme" and not os.path.exists(default_paths["extreme"]):
            print("⚠️ Extreme dataset missing, generating via real_data.py...")
            os.system("python data/mid_dimension/real_data.py")

        if data_source not in data_path_dict:
            raise ValueError(f"Unknown data_source: {data_source}")

        self.data_source = data_source
        self.data_path = data_path_dict[data_source]
        self.df = pd.read_csv(self.data_path, index_col=0)
        self.asset_dim = self.df.shape[1]
        self.data = self.df.values

        # === 2. Gym spaces ===
        self.window_size = window_size
        obs_dim = self.window_size * self.asset_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.asset_dim,), dtype=np.float32)

        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.portfolio_value = self.cash
        self.done = False
        return self._get_observation()

    def step(self, action):
        weights = np.clip(action, 0, 1).astype(np.float64)
        weights /= np.sum(weights) + 1e-8  # Normalize

        returns = self.data[self.current_step]
        portfolio_return = np.dot(weights, returns)
        reward = self.portfolio_value * portfolio_return

        self.portfolio_value += reward
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1

        obs = self._get_observation()
        info = {"portfolio_value": self.portfolio_value}
        return obs, reward, self.done, info

    def _get_observation(self):
        obs = self.data[self.current_step - self.window_size:self.current_step].flatten()
        return obs.astype(np.float32)




