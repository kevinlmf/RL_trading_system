import sys
import os

# === Add project root to Python path ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt

# === Import environments ===
from learning.env.copula_trading_env import CopulaTradingEnv
from learning.env.baseline_trading_env import BaselineTradingEnv

# === Import SAC agent ===
from learning.strategy.sac.agent import SACAgent

# === Helper functions for financial metrics ===
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

# === Hyperparameters ===
EPISODES = 500
WINDOW_SIZE = 30
INITIAL_CASH = 1e6
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Ensure output folders exist ===
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# === Define environments ===
environments = {
    "Copula": CopulaTradingEnv(data_source="extreme", window_size=WINDOW_SIZE, initial_cash=INITIAL_CASH),
    "Baseline": BaselineTradingEnv(data_source="extreme", window_size=WINDOW_SIZE, initial_cash=INITIAL_CASH)
}

# === Store rewards and metrics ===
reward_history = {env_name: [] for env_name in environments}
metrics = {}

# === SAC Training Loop ===
for env_name, env in environments.items():
    print(f"\n🚀 Training SAC in [{env_name}] environment...")

    # Auto detect state & action dim
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"✅ {env_name} | detected state_dim={state_dim}, action_dim={action_dim}")

    # Initialize SAC agent
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, device=DEVICE)

    cumulative_rewards = []  # Track cumulative portfolio value

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            reward_scaled = reward / INITIAL_CASH  # Normalize reward
            agent.replay_buffer.push(state, action, reward_scaled, next_state, done)

            state = next_state
            total_reward += reward_scaled

            # Update agent
            agent.update(batch_size=BATCH_SIZE)

        reward_history[env_name].append(total_reward)
        cumulative_rewards.append(info["portfolio_value"])

        if (ep + 1) % 10 == 0:
            print(f"[{env_name}] Episode {ep+1}/{EPISODES} | Total Reward: {total_reward:.6f}")

    # Save model
    model_path = f"models/sac_actor_{env_name.lower()}.pt"
    torch.save(agent.actor.state_dict(), model_path)
    print(f"✅ Saved model to {model_path}")

    # Compute metrics
    cumulative_returns = np.array(cumulative_rewards) / INITIAL_CASH - 1
    sharpe_ratio = calculate_sharpe_ratio(np.diff(cumulative_returns))
    max_drawdown = calculate_max_drawdown(cumulative_returns)
    annualized_return = calculate_annualized_return(cumulative_returns, EPISODES)

    metrics[env_name] = {
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Annualized Return": annualized_return
    }

# === Display Metrics Table ===
print("\n📊 Financial Metrics (SAC Copula vs Baseline):")
print(f"{'Environment':<10} | {'Sharpe':>6} | {'Max DD':>8} | {'Annualized Return':>18}")
print("-" * 50)
for env_name, m in metrics.items():
    print(f"{env_name:<10} | {m['Sharpe Ratio']:+.3f} | {m['Max Drawdown']:+.2%} | {m['Annualized Return']:+.2%}")

# === Plot Reward Comparison ===
plt.figure(figsize=(10, 6))
for env_name, rewards in reward_history.items():
    smooth_rewards = np.convolve(rewards, np.ones(10)/10, mode="valid")
    plt.plot(smooth_rewards, label=f"{env_name} SAC")

    # Annotate metrics
    m = metrics[env_name]
    plt.text(EPISODES * 0.7, smooth_rewards[-1],
             f"{env_name}\nSharpe={m['Sharpe Ratio']:.2f}\nMaxDD={m['Max Drawdown']:.1%}\nAnnual={m['Annualized Return']:.1%}",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightgrey", ec="black", lw=1))

plt.title("SAC Training: Copula vs Baseline")
plt.xlabel("Episode")
plt.ylabel("Smoothed Total Reward (normalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = "results/sac_copula_vs_baseline_metrics.png"
plt.savefig(plot_path)
plt.show()
print(f"📊 Saved reward comparison plot with metrics to {plot_path}")


