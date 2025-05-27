import os
import sys
import gymnasium as gym
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "env")))
from env.trading_env import TradingEnv
from env.data_loader import load_csv_data

# ✅ 加载数据
df = load_csv_data("data/SPY_1d.csv")

# ✅ 构建环境 + Monitor
env = TradingEnv(data=df, window_size=10)
env = Monitor(env)

# ✅ 时间戳 + 模型保存路径
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/ppo_trading_{timestamp}"
tensorboard_log_dir = f"tensorboard/ppo_run_{timestamp}"

# ✅ 创建 PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log=tensorboard_log_dir
)

# ✅ 训练前评估
mean_reward_before, _ = evaluate_policy(model, env, n_eval_episodes=5)
print(f"📊 Before training - Mean Reward: {mean_reward_before:.2f}")

# ✅ 开始训练
model.learn(total_timesteps=10000)

# ✅ 保存模型
os.makedirs("models", exist_ok=True)
model.save(model_path)
print(f"✅ PPO Model saved to {model_path}")

# ✅ 训练后评估
mean_reward_after, _ = evaluate_policy(model, env, n_eval_episodes=5)
print(f"📊 After training - Mean Reward: {mean_reward_after:.2f}")

