import gymnasium as gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, window_size=10, initial_balance=1000):
        super().__init__()
        self.df = data.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # 动作空间：0 - Hold，1 - Buy，2 - Sell
        self.action_space = gym.spaces.Discrete(3)

        # 观测空间：窗口长度 × 特征数（如 OHLCV）
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.df.shape[1]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):  # ✅ 兼容 Gymnasium + SB3
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: flat, 1: long
        self.position_price = 0
        self.current_step = self.window_size
        self.done = False

        return self._get_observation(), {}  # ✅ 返回 obs, info

    def _get_observation(self):
        return self.df.iloc[self.current_step - self.window_size:self.current_step].values

    def step(self, action):
        price = self.df.iloc[self.current_step]['Close']
        reward = 0

        # 动作逻辑
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.position_price = price
            elif self.position == -1:
                reward = self.position_price - price
                self.balance += reward
                self.position = 0

        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.position_price = price
            elif self.position == 1:
                reward = price - self.position_price
                self.balance += reward
                self.position = 0

        # 奖励 shaping：惩罚频繁交易
        shaping = -0.01 if action != 0 else 0
        reward += shaping

        # 时间推进
        self.current_step += 1
        self.done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()

        return obs, reward, self.done, False, {}  # ✅ SB3 兼容格式：obs, reward, terminated, truncated, info

