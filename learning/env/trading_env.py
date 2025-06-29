import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, window_size, initial_balance=10000, reward_fn=None, price_column=None):
        super().__init__()
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reward_fn = reward_fn
        self.price_column = price_column or df.columns[0]  # 默认使用第1列作为价格列

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, df.shape[1]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        self.current_step = self.window_size
        self.done = False

        obs = self._get_observation()
        return obs, {}  # ✅ gym>=0.26 API: (obs, info)

    def _get_observation(self):
        obs = self.df.iloc[self.current_step - self.window_size : self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        price = self.df.iloc[self.current_step][self.price_column]
        prev_value = self.balance + self.shares_held * price

        # === Execute action ===
        if action == 1:  # Buy
            n_shares = self.balance // price
            self.shares_held += n_shares
            self.balance -= n_shares * price
        elif action == 2:  # Sell
            self.balance += self.shares_held * price
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        self.done = done

        new_value = self.balance + self.shares_held * price
        reward = (new_value - prev_value) / prev_value if prev_value != 0 else 0

        self.total_value = new_value
        obs = self._get_observation()
        info = {"portfolio_value": self.total_value}

        return obs, reward, done, False, info  # ✅ gym>=0.26 API: obs, reward, terminated, truncated, info

