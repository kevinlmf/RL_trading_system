import sys
import os
sys.path.append(os.path.abspath("."))  # ✅ 解决 import learning.* 问题

import pandas as pd
import torch
from learning.env.trading_env import TradingEnv
from learning.strategy.rl.ppo.ppo_agent import PPOAgent

# === Config ===
DATA_PATH = "data/low_dimension/simulated_copula_data.csv"
MODEL_LOAD_PATH = "ppo_actor_critic.pt"
WINDOW_SIZE = 30
INITIAL_BALANCE = 10000

# === Load env & model ===
print(f"📂 Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
env = TradingEnv(df, window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE)

obs_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = PPOAgent(obs_shape=obs_shape, n_actions=n_actions)  # ✅ 修正参数名

# === Load model weights ===
agent.model.load_state_dict(torch.load(MODEL_LOAD_PATH))
agent.model.eval()
print(f"📥 Loaded model from {MODEL_LOAD_PATH}")

# === Evaluate one episode ===
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _, _ = agent.act(obs, evaluation=True)
    obs, reward, terminated, _, _ = env.step(action)
    total_reward += reward
    done = terminated

print(f"\n🎯 Evaluation Total Reward = {total_reward:.2f}")


