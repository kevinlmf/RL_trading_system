import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.trading_env import TradingEnv
from env.data_loader import load_csv_data  


data = load_csv_data("data/SPY_1d.csv")


env = TradingEnv(data=data)

n_episodes = 3

for ep in range(n_episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  
        obs, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward

    print(f"Episode {ep+1} - Total Reward: {total_reward}")
