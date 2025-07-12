import sys
import os

# === Add project root to Python path ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# === Import environments ===
from learning.env.copula_trading_env import CopulaTradingEnv
from learning.env.baseline_trading_env import BaselineTradingEnv

# === Import agents ===
from learning.strategy.ppo.ppo_agent import PPOAgent as PPO
from learning.strategy.cppo.agent import CPPOAgent as CPPO
from learning.strategy.sac.agent import SACAgent as SAC

# === Helper functions ===
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = np.array(returns) - risk_free_rate
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)

def calculate_max_drawdown(cumulative_returns):
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (peak + 1e-8)
    return np.min(drawdown)

def calculate_annualized_return(cumulative_returns, episodes, scale=252):
    total_return = cumulative_returns[-1]
    return (1 + total_return) ** (scale / episodes) - 1

# === Training loop ===
def train_agent(agent_class, env, episodes, batch_size, initial_cash, device):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"✅ Training {agent_class.__name__} | state_dim={state_dim}, action_dim={action_dim}")

    agent = agent_class(state_dim=state_dim, action_dim=action_dim, device=device)
    cumulative_rewards = []
    reward_history = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            reward_scaled = reward / initial_cash
            if hasattr(agent, "replay_buffer"):
                agent.replay_buffer.push(state, action, reward_scaled, next_state, done)
            else:
                agent.store_transition((state, action, reward_scaled, done, 0.0, 0.0))  # PPO style

            state = next_state
            total_reward += reward_scaled

            if hasattr(agent, "update"):
                agent.update(batch_size=batch_size)

        reward_history.append(total_reward)
        cumulative_rewards.append(info["portfolio_value"])

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward:.6f}")

    return reward_history, cumulative_rewards

# === Main pipeline ===
def main(args):
    # Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Environments
    environments = {
        "Copula": CopulaTradingEnv(data_source=args.data_source, window_size=args.window_size, initial_cash=args.initial_cash),
        "Baseline": BaselineTradingEnv(data_source=args.data_source, window_size=args.window_size, initial_cash=args.initial_cash)
    }

    # Agent mapping
    agents = {"ppo": PPO, "cppo": CPPO, "sac": SAC}
    if args.agent.lower() not in agents:
        raise ValueError(f"Unsupported agent: {args.agent}. Choose from ppo, cppo, sac")

    agent_class = agents[args.agent.lower()]
    metrics = {}

    # Train agents in each environment
    for env_name, env in environments.items():
        print(f"\n🚀 [{args.agent.upper()}] Training in {env_name} environment...")
        rewards, portfolio_values = train_agent(agent_class, env, args.episodes, args.batch_size, args.initial_cash, DEVICE)

        # Compute metrics
        cumulative_returns = np.array(portfolio_values) / args.initial_cash - 1
        metrics[env_name] = {
            "Sharpe Ratio": calculate_sharpe_ratio(np.diff(cumulative_returns)),
            "Max Drawdown": calculate_max_drawdown(cumulative_returns),
            "Annualized Return": calculate_annualized_return(cumulative_returns, args.episodes)
        }

        # Save smoothed rewards
        smooth_rewards = np.convolve(rewards, np.ones(10)/10, mode="valid")
        plt.plot(smooth_rewards, label=f"{env_name} ({args.agent.upper()})")

    # Print Metrics Table
    print("\n📊 Financial Metrics:")
    print(f"{'Environment':<10} | {'Sharpe':>6} | {'Max DD':>8} | {'Annualized Return':>18}")
    print("-" * 50)
    for env_name, m in metrics.items():
        print(f"{env_name:<10} | {m['Sharpe Ratio']:+.3f} | {m['Max Drawdown']:+.2%} | {m['Annualized Return']:+.2%}")

    # Plot
    plt.title(f"{args.agent.upper()} Training: Copula vs Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward (normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"results/{args.agent.lower()}_{args.data_source}_comparison.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"📊 Saved reward comparison plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="sac", help="Agent to train: ppo, cppo, sac")
    parser.add_argument("--data-source", type=str, default="real", help="Data source: real, crypto, forex, simulated")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--window-size", type=int, default=30, help="Observation window size")
    parser.add_argument("--initial-cash", type=float, default=1e6, help="Initial portfolio value")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    args = parser.parse_args()
    main(args)
