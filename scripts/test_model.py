import os
import sys
import time
import argparse

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.trading_env import TradingEnv
from env.data_loader import load_csv_data

def load_model(model_path, algo):
    if algo == "ppo":
        return PPO.load(model_path)
    elif algo == "dqn":
        return DQN.load(model_path)
    else:
        raise ValueError("Unsupported algorithm. Use 'ppo' or 'dqn'.")

def test_model(model, env, n_episodes=3):
    print(f"🎯 Running {n_episodes} test episodes...")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            time.sleep(0.01)

        print(f"✅ Episode {ep+1} - Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (without .zip)")
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], required=True, help="Algorithm: ppo or dqn")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
    args = parser.parse_args()

    # 加载数据
    data = load_csv_data("data/SPY_1d.csv")
    env = TradingEnv(data=data)

    # 加载模型
    model = load_model(args.model, args.algo)

    # 执行测试
    test_model(model, env, n_episodes=args.episodes)
