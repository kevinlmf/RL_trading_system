import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.trading_env import TradingEnv
from env.data_loader import load_csv_data  

data = load_csv_data("data/SPY_1d.csv")
env = TradingEnv(data=data)

n_episodes = 3

for ep in range(n_episodes):
    obs, _ = env.reset()  # ✅ 新版 Gymnasium reset 返回 obs, info
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  
        obs, reward, done, _, _ = env.step(action)  # ✅ Gymnasium 返回 5 个值
        # env.render()  # 如无定义可注释
        total_reward += reward
        time.sleep(0.01)  # 更平滑输出（可选）

    print(f"🎲 Episode {ep+1} - Total Reward: {total_reward:.2f}")


