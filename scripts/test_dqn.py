import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.trading_env import TradingEnv
from env.data_loader import load_csv_data

# 加载数据
data = load_csv_data("data/SPY_1d.csv")
env = TradingEnv(data=data)

# 加载训练好的模型
model = DQN.load("models/dqn_trading")

# 推理模式
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()
    total_reward += reward

print(f"✅ Total evaluation reward: {total_reward:.2f}")
