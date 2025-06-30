import torch
from learning.env.trading_env import TradingEnv
from learning.strategy.dqn.dqn_agent import DQNAgent
from learning.strategy.dqn.dqn_network import QNetwork



EPISODES = 50
WINDOW_SIZE = 30
INITIAL_CASH = 1e6
BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 10

env = TradingEnv(data_source="simulated", window_size=WINDOW_SIZE, initial_cash=INITIAL_CASH)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

q_network = QNetwork(state_dim, action_dim)
agent = DQNAgent(q_network=q_network,
                 state_dim=state_dim,
                 action_dim=action_dim,
                 batch_size=BATCH_SIZE)

for ep in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action_idx = agent.select_action(state)
        action = torch.nn.functional.one_hot(torch.tensor(action_idx), num_classes=action_dim).numpy()
        next_state, reward, done, info = env.step(action)

        agent.store_transition(state, action_idx, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += reward

    if ep % UPDATE_TARGET_EVERY == 0:
        agent.update_target_network()

    print(f"Episode {ep+1}/{EPISODES} | Total Reward: {total_reward:.2f}")

torch.save(q_network.state_dict(), "models/dqn_q_network.pt")




