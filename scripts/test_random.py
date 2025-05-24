import os
import sys
import gymnasium as gym
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.trading_env import TradingEnv
from env.data_loader import load_csv_data

# 加载数据
data = load_csv_data("data/SPY_1d.csv")
env = TradingEnv(data=data)

# 重置环境
obs, _ = env.reset()
done = False
total_reward = 0
balances = []

# 随机策略推理 loop
while not done:
    action = env.action_space.sample()  # ✅ 随机动作
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()
    balances.append(env.balance)
    total_reward += reward

print(f"\n🎲 Random Strategy Total evaluation reward: {total_reward:.2f}")

