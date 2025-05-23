import numpy as np
import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data=None, window_size=10, initial_balance=1000):
        super().__init__()
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.data = data if data is not None else self._generate_synthetic_data()
        self.current_step = self.window_size

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 5), dtype=np.float32)

        self.reset()

    def _generate_synthetic_data(self):
        prices = np.cumsum(np.random.randn(1000)) + 100
        ohlcv = np.column_stack([prices, prices+1, prices-1, prices, np.random.rand(1000)*100])
        return ohlcv

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = self.window_size
        return self._get_observation()

    def _get_observation(self):
        return self.data[self.current_step - self.window_size:self.current_step]

    def step(self, action):
        price = self.data[self.current_step][3]
        reward = 0

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price
        elif action == 0 and self.position != 0:
            pnl = (price - self.entry_price) * self.position
            reward += pnl
            self.balance += pnl
            self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")
