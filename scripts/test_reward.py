import numpy as np
import sys
import os

# Add parent directory to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from learning.env.trading_env import TradingEnv
from exploration.latent_bonus import LatentFactorBonus

# === Initialize Trading Environment ===
env = TradingEnv(data_source="simulated", window_size=30)
obs = env.reset()
print("🧩 Initial Observation Shape:", obs.shape)

# === Initialize Exploration Bonus Module ===
bonus_fn = LatentFactorBonus(n_components=2, beta=0.05, bandwidth=0.3)

# === Initialize Statistics ===
total_reward = 0.0
total_bonus = 0.0
step_count = 0

# === Simulate One Episode ===
done = False
while not done:
    action = np.random.uniform(-1, 1, size=env.asset_dim)  # Random action (replaceable with learned policy)
    
    obs, reward, done, info = env.step(action)
    
    bonus_fn.update_memory(obs)
    if step_count % 10 == 0:  # Update KDE every 10 steps
        bonus_fn.fit_latent_space()

    bonus = bonus_fn.compute_bonus(obs)
    total_reward += reward
    total_bonus += bonus
    step_count += 1

    print(f"Step {step_count:03d} | Reward: {reward:.2f} | Bonus: {bonus:.2f} | PV: {info['portfolio_value']:.2f}")

# === Summary ===
print("\n✅ Episode finished.")
print(f"📈 Final Portfolio Value: {info['portfolio_value']:.2f}")
print(f"🎯 Total Reward: {total_reward:.2f}")
print(f"✨ Total Exploration Bonus: {total_bonus:.2f}")
