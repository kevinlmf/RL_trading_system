import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from learning.env.trading_env import TradingEnv
from learning.strategy.rl.ppo.ppo_agent import PPOAgent
from learning.exploration.latent_bonus import LatentFactorBonus

# === Config ===
DATA_PATH = "data/low_dimension/simulated_copula_data.csv"
WINDOW_SIZE = 30
INITIAL_BALANCE = 10000
EPISODES = 10
STEPS_PER_UPDATE = 2048
MODEL_SAVE_PATH = "ppo_actor_critic.pt"

# === Load data & create env ===
print(f"📂 Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
env = TradingEnv(df, window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE)

obs_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = PPOAgent(obs_shape=obs_shape, n_actions=n_actions)
latent_bonus = LatentFactorBonus(n_components=2, beta=0.2, bandwidth=0.3)

print(f"🚀 Starting PPO training for {EPISODES} episodes...\n")

for episode in range(EPISODES):
    obs = env.reset()[0]
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        action, logprob = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        latent_bonus.update_memory(obs)
        if step_count % 100 == 0:
            latent_bonus.fit_latent_space()
        bonus = latent_bonus.compute_bonus(obs)
        reward += bonus

        agent.store(obs, action, reward, logprob, done)
        obs = next_obs
        total_reward += reward
        step_count += 1

        if step_count % STEPS_PER_UPDATE == 0:
            agent.train()

    print(f"✅ Episode {episode+1}/{EPISODES} | Total Reward = {total_reward:.2f}")

# === Save model ===
torch.save(agent.actor.state_dict(), MODEL_SAVE_PATH)
print(f"\n📦 PPO model saved to: {MODEL_SAVE_PATH}")






