import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from env.trading_env import TradingEnv
from env.data_loader import load_csv_data

df = load_csv_data("data/SPY_1d.csv")
env = TradingEnv(data=df, window_size=10)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.trading_env import TradingEnv
from env.data_loader import load_csv_data

# 加载数据
data = load_csv_data("data/SPY_1d.csv")

# 构建环境 + Monitor（自动记录 reward/length）
env = TradingEnv(data=data)
env = Monitor(env)

# 构建 DQN 模型
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=1,
    target_update_interval=500,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

# 训练模型
model.learn(total_timesteps=10000)

# 保存模型
os.makedirs("models", exist_ok=True)
model.save("models/dqn_trading")
print("✅ Model saved to models/dqn_trading")

