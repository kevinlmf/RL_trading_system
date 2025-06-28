import os
import sys
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from learning.env.trading_env import TradingEnv
from learning.strategy.rl.dqn.dqn_agent import DQNAgent
from learning.strategy.rl.ppo.ppo_agent import PPOAgent
from learning.strategy.rl.random.random_policy import RandomPolicy


# === Config ===
DATA_PATH = "data/low_dimension/simulated_copula_data.csv"
WINDOW_SIZE = 30
INITIAL_BALANCE = 10000
PPO_MODEL_PATH = "ppo_actor_critic.pt"
DQN_MODEL_PATH = "dqn_model.pt"
EPISODES = 5

def compute_sharpe_ratio(returns):
    returns = np.array(returns)
    if returns.std() == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)

def compute_max_drawdown(values):
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return np.max(drawdown)

def evaluate(agent, env, name):
    all_rewards = []
    all_portfolios = []

    for episode in range(EPISODES):
        obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        done = False
        rewards = []
        values = []

        while not done:
            action = agent.select_action(obs) if hasattr(agent, "select_action") else agent.act(obs)
            if isinstance(action, tuple):  # PPO returns (action, log_prob)
                action = action[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            values.append(info.get("portfolio_value", env.balance))

        all_rewards.append(np.sum(rewards))
        all_portfolios.append(values)

    avg_reward = np.mean(all_rewards)
    sharpe = compute_sharpe_ratio(np.diff(all_portfolios[-1]) / np.array(all_portfolios[-1][:-1]))
    max_dd = compute_max_drawdown(all_portfolios[-1])

    print(f"\n📊 {name} Strategy (averaged over {EPISODES} episode(s)):")
    print(f" - Total Reward     = {avg_reward:.2f}")
    print(f" - Sharpe Ratio     = {sharpe:.4f}")
    print(f" - Max Drawdown     = {max_dd:.4f}")

    return all_portfolios[-1]

def plot_portfolios(portfolios):
    plt.figure(figsize=(10, 6))
    for name, series in portfolios.items():
        plt.plot(series, label=name)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("evaluation", exist_ok=True)
    plt.savefig("evaluation/portfolio_comparison_full.png")
    plt.show()

if __name__ == "__main__":
    print(f"📂 Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    obs_shape = (WINDOW_SIZE, df.shape[1])
    n_actions = 3  # Buy / Hold / Sell

    portfolios = {}

    # === Evaluate DQN ===
    env_dqn = TradingEnv(df, window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE)
    agent_dqn = DQNAgent(obs_shape=obs_shape, n_actions=n_actions)
    agent_dqn.model.load_state_dict(torch.load(DQN_MODEL_PATH))
    agent_dqn.model.eval()
    portfolios["DQN"] = evaluate(agent_dqn, env_dqn, name="DQN")

    # === Evaluate PPO ===
    env_ppo = TradingEnv(df, window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE)
    agent_ppo = PPOAgent(obs_shape=obs_shape, action_dim=n_actions)
    agent_ppo.model.load_state_dict(torch.load(PPO_MODEL_PATH))
    agent_ppo.model.eval()
    portfolios["PPO"] = evaluate(agent_ppo, env_ppo, name="PPO")

    # === Evaluate Random ===
    env_rand = TradingEnv(df, window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE)
    agent_rand = RandomPolicy(env_rand.action_space)
    portfolios["Random"] = evaluate(agent_rand, env_rand, name="Random")

    # === Plot all results ===
    plot_portfolios(portfolios)




