import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.trading_env import TradingEnv
from env.data_loader import load_csv_data

# 加载数据
data = load_csv_data("data/SPY_1d.csv")
env = TradingEnv(data=data)

# 加载 PPO 模型
model = PPO.load("models/ppo_trading")

# 推理评估
obs, _ = env.reset()
done = False
total_reward = 0
balances = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()
    balances.append(env.balance)
    total_reward += reward

print(f"\n✅ PPO Total evaluation reward: {total_reward:.2f}")

