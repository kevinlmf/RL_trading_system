import numpy as np
import gymnasium as gym                      
from gymnasium import spaces                

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

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = self.window_size
        return self._get_observation(), {}

    def _get_observation(self):
        return self.data[self.current_step - self.window_size:self.current_step]

    def step(self, action):
        price = self.data[self.current_step][3]  # Close price
        reward = 0
        cost = 0

        # === 开仓 ===
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
            cost = 0.001 * price  # 手续费

        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price
            cost = 0.001 * price

        # === 平仓 ===
        elif action == 0 and self.position != 0:
            pnl = (price - self.entry_price) * self.position
            reward += pnl
            self.balance += pnl
            cost = 0.001 * price
            self.position = 0

        # === 手续费 & 状态更新 ===
        reward -= cost
        self.balance -= cost
        self.current_step += 1

        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")

