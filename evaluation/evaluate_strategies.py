import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from learning.env.trading_env import TradingEnv
from learning.strategy.ppo.ppo_agent import PPOAgent
from learning.strategy.ppo.ppo_network import ActorCritic
from learning.strategy.random_strategy import RandomStrategy
from learning.strategy.buy_and_hold_strategy import BuyAndHoldStrategy


def compute_metrics(rewards):
    cumulative_returns = np.cumsum(rewards)
    sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-8)
    max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
    return {
        "Total Reward": round(cumulative_returns[-1], 2),
        "Sharpe Ratio": round(sharpe_ratio, 3),
        "Max Drawdown": round(max_drawdown, 2)
    }


def evaluate_agent(agent, env):
    state = env.reset()
    rewards = []
    done = False
    while not done:
        if isinstance(agent, PPOAgent):
            action, _ = agent.select_action(state)
        else:
            action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
    return compute_metrics(rewards)


if __name__ == "__main__":
    env = TradingEnv(data_source="mid_dimension", window_size=30)

    # Use true dimension used in training (hardcoded here as 990)
    trained_obs_dim = 990
    action_dim = env.action_space.shape[0]

    # Random Strategy
    print("\n📊 Random Strategy:")
    random_agent = RandomStrategy(action_dim)
    random_metrics = evaluate_agent(random_agent, env)
    print(random_metrics)

    # Buy and Hold
    print("\n📊 Buy and Hold Strategy:")
    buy_and_hold = BuyAndHoldStrategy(env)
    buy_and_hold_metrics = evaluate_agent(buy_and_hold, env)
    print(buy_and_hold_metrics)

    # PPO Strategy
    print("\n📊 PPO Strategy:")
    actor_critic = ActorCritic(state_dim=trained_obs_dim, action_dim=action_dim)
    actor_critic.load_state_dict(torch.load("models/ppo_actor_critic.pt", map_location="cpu"))
    ppo_agent = PPOAgent(actor_critic=actor_critic, device="cpu")
    ppo_metrics = evaluate_agent(ppo_agent, env)
    print(ppo_metrics)

    # Comparison
    strategies = ["Random", "Buy&Hold", "PPO"]
    rewards = [random_metrics["Total Reward"], buy_and_hold_metrics["Total Reward"], ppo_metrics["Total Reward"]]

    plt.bar(strategies, rewards)
    plt.title("Total Reward Comparison")
    plt.ylabel("Total Reward")
    plt.savefig("evaluation/portfolio_comparison_full.png")
    plt.show()