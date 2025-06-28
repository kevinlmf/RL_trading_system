

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from learning.env.trading_env import TradingEnv
from learning.strategy.rl.dqn.dqn_agent import DQNAgent

# === Config ===
DATA_PATH = "data/low_dimension/simulated_copula_data.csv"
WINDOW_SIZE = 30
INITIAL_BALANCE = 10000
MODEL_PATH = "dqn_model.pt"

# === Load data & create env ===
print(f"📂 Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
env = TradingEnv(df, window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE)

obs_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions)

# === Load trained model ===
agent.model.load_state_dict(torch.load(MODEL_PATH))
agent.model.eval()
print(f"📥 Loaded model from {MODEL_PATH}")

# === Run one evaluation episode ===
reset_result = env.reset()
obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
total_reward = 0
done = False

while not done:
    action = agent.act(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"\n🎯 Evaluation Total Reward = {total_reward:.2f}")


