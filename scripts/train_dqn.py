import torch
import pandas as pd
from learning.env.trading_env import TradingEnv
from learning.strategy.rl.dqn.dqn_agent import DQNAgent

# === Load data ===
df = pd.read_csv("data/low_dimension/simulated_copula_data.csv")
env = TradingEnv(df, window_size=10)

# === Init agent ===
obs_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = DQNAgent(obs_shape, n_actions)

# === Training ===
n_episodes = 50

for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.remember(obs, action, reward, next_obs, done)
        agent.learn()

        obs = next_obs
        total_reward += reward

    agent.update_target()
    print(f"📘 Episode {episode + 1} | Total Reward: {total_reward:.2f}")

# === Save model ===
import os
os.makedirs("models", exist_ok=True)
torch.save(agent.model.state_dict(), "models/dqn_model.pt")
print("📦 DQN model saved to: models/dqn_model.pt")









