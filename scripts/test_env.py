from learning.env.trading_env import TradingEnv
import numpy as np

# === 创建环境（使用 simulation 数据） ===
env = TradingEnv(data_source="simulated")

obs = env.reset()
print("🧩 Initial Observation Shape:", obs.shape)

done = False
total_reward = 0

while not done:
    action = np.random.uniform(-1, 1, size=env.asset_dim)  # 随机策略
    obs, reward, done, info = env.step(action)
    total_reward += reward

print("✅ Episode finished. Final portfolio value:", info["portfolio_value"])
print("📈 Total reward:", total_reward)
