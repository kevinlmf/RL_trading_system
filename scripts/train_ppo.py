import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.trading_env import TradingEnv
from env.data_loader import load_csv_data

# 加载数据
data = load_csv_data("data/SPY_1d.csv")

# 构建环境
env = TradingEnv(data=data)
env = Monitor(env)

# 构建 PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

# 开始训练
model.learn(total_timesteps=20000)

# 保存模型
os.makedirs("models", exist_ok=True)
model.save("models/ppo_trading")
print("✅ PPO model saved to models/ppo_trading")
