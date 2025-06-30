import gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A multi-asset trading environment for reinforcement learning.
    It supports both real and simulated data and allows flexible reward shaping.
    """

    def __init__(self, 
                 data_source="simulated", 
                 data_path_dict=None,
                 window_size=30,
                 initial_cash=1e6,
                 reward_fn=None):
        super(TradingEnv, self).__init__()

        # === 1. Select the data source (simulated, real, etc.) ===
        default_paths = {
            "simulated": "data/mid_dimension/simulated_copula_returns.csv",
            "real": "data/mid_dimension/real_asset_log_returns.csv",
            "mid_dimension": "data/mid_dimension/real_asset_log_returns.csv"  # optional alias
        }

        if data_path_dict is None:
            data_path_dict = default_paths

        if data_source not in data_path_dict:
            raise ValueError(f"Unknown data_source: {data_source}")
        
        self.data_source = data_source
        self.data_path = data_path_dict[data_source]
        self.df = pd.read_csv(self.data_path)
        self.asset_dim = self.df.shape[1]
        self.data = self.df.values

        # === 2. Initialize environment parameters ===
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.reward_fn = reward_fn
        self.reset()

        # === 3. Define Gym-compatible observation and action spaces ===
        obs_dim = self.window_size * self.asset_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.asset_dim,), dtype=np.float32)

    def reset(self):
        """
        Reset the environment to initial state.
        """
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.portfolio_value = self.cash
        self.prev_portfolio_value = self.portfolio_value

        if self.reward_fn and hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()

        return self._get_observation()

    def step(self, action):
        """
        Execute one time step in the environment.
        """
        assert self.current_step < len(self.data) - 1

        # === 1. Clip and normalize action to get portfolio weights ===
        weights = np.clip(action, -1, 1)
        weights = weights / (np.sum(np.abs(weights)) + 1e-8)

        # === 2. Compute daily portfolio return ===
        returns = self.data[self.current_step]
        portfolio_return = np.dot(weights, returns)
        reward_daily = self.portfolio_value * portfolio_return

        # === 3. Update portfolio value ===
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value += reward_daily
        self.cash = self.portfolio_value

        # === 4. Compute reward using optional custom function ===
        obs = self._get_observation()
        if self.reward_fn:
            reward = self.reward_fn(obs=obs, reward=reward_daily)
        else:
            reward = reward_daily

        # === 5. Determine episode termination ===
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        info = {
            "portfolio_value": self.portfolio_value,
            "return": portfolio_return
        }
        return obs, reward, done, info

    def _get_observation(self):
        """
        Return the flattened historical price window as the observation.
        """
        window_data = self.data[self.current_step - self.window_size:self.current_step]
        obs = window_data.flatten()
        return obs.astype(np.float32)

