import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from learning.env.trading_env import TradingEnv
from learning.strategy.rl.dqn.dqn_agent import DQNAgent
from exploration.latent_bonus import LatentFactorBonus


# === Config ===
DATA_PATH = "data/low_dimension/simulated_copula_data.csv"
WINDOW_SIZE = 30
INITIAL_BALANCE = 10000
EPISODES = 10
MODEL_SAVE_PATH = "dqn_model.pt"

# === Load data & create env ===
print(f"📂 Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
env = TradingEnv(df, window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE)

obs_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions)
latent_bonus = LatentFactorBonus(n_components=2, beta=0.2, bandwidth=0.3)

print(f"🚀 Starting DQN training for {EPISODES} episodes...\n")

for episode in range(EPISODES):
    obs = env.reset()[0]  # unpack if (obs, info) tuple
    total_reward = 0
    done = False

    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # === Latent Exploration Bonus ===
        latent_bonus.update_memory(obs)
        if episode % 1 == 0:
            latent_bonus.fit_latent_space()
        bonus = latent_bonus.compute_bonus(obs)
        reward += bonus

        agent.store(obs, action, reward, next_obs, done)
        agent.train_step()
        obs = next_obs
        total_reward += reward

    print(f"✅ Episode {episode+1}/{EPISODES} | Total Reward = {total_reward:.2f}")

# === Save model ===
torch.save(agent.model.state_dict(), MODEL_SAVE_PATH)
print(f"\n📦 DQN model saved to: {MODEL_SAVE_PATH}")
