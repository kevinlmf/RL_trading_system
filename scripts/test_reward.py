import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from learning.env.trading_env import TradingEnv
from exploration.latent_bonus import LatentFactorBonus

# === 初始化环境 ===
env = TradingEnv(data_source="simulated", window_size=30)
obs = env.reset()
print("🧩 Initial Observation Shape:", obs.shape)

# === 初始化探索奖励模块 ===
bonus_fn = LatentFactorBonus(n_components=2, beta=0.05, bandwidth=0.3)

# === 初始化统计量 ===
total_reward = 0.0
total_bonus = 0.0
step_count = 0

# === 模拟一个 episode ===
done = False
while not done:
    action = np.random.uniform(-1, 1, size=env.asset_dim)  # 随机动作（也可以换成其他策略）
    
    obs, reward, done, info = env.step(action)
    
    bonus_fn.update_memory(obs)
    if step_count % 10 == 0:  # 每 10 步更新一次 KDE
        bonus_fn.fit_latent_space()

    bonus = bonus_fn.compute_bonus(obs)
    total_reward += reward
    total_bonus += bonus
    step_count += 1

    print(f"Step {step_count:03d} | Reward: {reward:.2f} | Bonus: {bonus:.2f} | PV: {info['portfolio_value']:.2f}")

# === 总结 ===
print("\n✅ Episode finished.")
print(f"📈 Final Portfolio Value: {info['portfolio_value']:.2f}")
print(f"🎯 Total Reward: {total_reward:.2f}")
print(f"✨ Total Exploration Bonus: {total_bonus:.2f}")

