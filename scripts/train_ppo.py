import torch
from learning.env.trading_env import TradingEnv
from learning.strategy.ppo.ppo_agent import PPOAgent
from learning.strategy.ppo.ppo_network import ActorCritic

# === Hyperparameters ===
EPISODES = 50
WINDOW_SIZE = 30
INITIAL_CASH = 1e6
BATCH_SIZE = 64
DEVICE = "cpu"  

# === Initialize Environment ===
env = TradingEnv(data_source="simulated", window_size=WINDOW_SIZE, initial_cash=INITIAL_CASH)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# === Initialize PPO Agent ===
model = ActorCritic(state_dim, action_dim).to(DEVICE)
agent = PPOAgent(actor_critic=model, batch_size=BATCH_SIZE, device=DEVICE)

# === Training Loop ===
for ep in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_idx, log_prob = agent.select_action(state)
        action = torch.nn.functional.one_hot(torch.tensor(action_idx), num_classes=action_dim).numpy()
        next_state, reward, done, info = env.step(action)

        with torch.no_grad():
            _, value = model(torch.tensor(state, dtype=torch.float32).to(DEVICE))

        agent.store_transition((state, action_idx, reward, done, log_prob, value.item()))

        state = next_state
        total_reward += reward

    agent.train_step()
    print(f"Episode {ep+1}/{EPISODES} | Total Reward: {total_reward:.2f}")

# === Save Trained Model ===
torch.save(model.state_dict(), "models/ppo_actor_critic.pt")











