import pandas as pd
import numpy as np
import torch
from learning.env.trading_env import TradingEnv
from learning.strategy.rl.dqn.dqn_agent import DQNAgent
from learning.strategy.rl.ppo.ppo_agent import PPOAgent

# === Load data ===
df = pd.read_csv("data/low_dimension/simulated_copula_data.csv")
env = TradingEnv(df, window_size=10)
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

# === Load models ===
DQN_MODEL_PATH = "models/dqn_model.pt"
PPO_MODEL_PATH = "models/ppo_actor_critic.pt"

agent_dqn = DQNAgent(obs_shape, n_actions)
agent_dqn.model.load_state_dict(torch.load(DQN_MODEL_PATH))

agent_ppo = PPOAgent(obs_shape, n_actions)
agent_ppo.model.load_state_dict(torch.load(PPO_MODEL_PATH))


def evaluate_agent(env, agent, is_ppo=False):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    values = []

    while not done:
        if is_ppo:
            action, _, _ = agent.select_action(obs)
        else:
            action = agent.select_action(obs, epsilon=0.0)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        values.append(info["portfolio_value"])

    return total_reward, values


def evaluate_baseline(env, strategy_name="buy_and_hold"):
    obs, _ = env.reset()
    total_reward = 0
    values = []

    if strategy_name == "buy_and_hold":
        action_plan = [1] + [0] * (len(env.df) - env.window_size - 2) + [2]
    elif strategy_name == "random":
        np.random.seed(42)
        action_plan = np.random.choice([0, 1, 2], size=len(env.df) - env.window_size)
    else:
        raise ValueError("Unknown strategy name.")

    for action in action_plan:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        values.append(info["portfolio_value"])
        if terminated or truncated:
            break

    return total_reward, values


def compute_metrics(values):
    values = np.array(values)
    returns = np.diff(values) / values[:-1]
    cumulative_return = (values[-1] - values[0]) / values[0] * 100
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
    drawdown = 1 - values / np.maximum.accumulate(values)
    max_drawdown = np.max(drawdown) * 100
    volatility = np.std(returns) * 100
    return cumulative_return, sharpe_ratio, max_drawdown, volatility


def print_results(name, reward, values):
    cr, sr, dd, vol = compute_metrics(values)
    print(f"\n📊 {name} Strategy:")
    print(f" - Total Reward       = {reward:.2f}")
    print(f" - Cumulative Return  = {cr:.2f}%")
    print(f" - Sharpe Ratio       = {sr:.2f}")
    print(f" - Max Drawdown       = {dd:.2f}%")
    print(f" - Volatility         = {vol:.2f}%")


# === Evaluate All ===
print_results("DQN", *evaluate_agent(TradingEnv(df, 10), agent_dqn, is_ppo=False))
print_results("PPO", *evaluate_agent(TradingEnv(df, 10), agent_ppo, is_ppo=True))
print_results("Buy-and-Hold", *evaluate_baseline(TradingEnv(df, 10), "buy_and_hold"))
print_results("Random", *evaluate_baseline(TradingEnv(df, 10), "random"))


