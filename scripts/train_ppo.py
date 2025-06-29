import torch
import pandas as pd
from learning.env.trading_env import TradingEnv
from learning.strategy.rl.ppo.ppo_agent import PPOAgent

# === Load data ===
df = pd.read_csv("data/low_dimension/simulated_copula_data.csv")
env = TradingEnv(df, window_size=10)

# === Init agent ===
obs_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = PPOAgent(obs_shape, n_actions)

# === Training ===
n_episodes = 50

for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    trajectory = []

    while not done:
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        trajectory.append((obs, action, log_prob, reward, value))
        obs = next_obs
        total_reward += reward

    agent.learn(trajectory)
    print(f"📘 Episode {episode + 1} | Total Reward: {total_reward:.2f}")

# === Save model ===
import os
os.makedirs("models", exist_ok=True)
torch.save(agent.model.state_dict(), "models/ppo_actor_critic.pt")
print("📦 PPO model saved to: models/ppo_actor_critic.pt")


