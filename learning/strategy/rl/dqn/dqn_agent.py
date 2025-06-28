import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from learning.strategy.rl.dqn.dqn_network import DQNNet

class DQNAgent:
    def __init__(self, obs_shape, n_actions, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.model = DQNNet(obs_shape, n_actions).to(self.device)
        self.target_model = DQNNet(obs_shape, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.n_actions = n_actions

    def act(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        return torch.argmax(q_values).item()

    def select_action(self, obs):  # ✅ Add this for evaluation compatibility
        return self.act(obs)

    def store(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        obs, action, reward, next_obs, done = zip(*batch)

        obs = np.array([o.flatten() for o in obs])
        next_obs = np.array([no.flatten() for no in next_obs])

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        action = torch.tensor(action).unsqueeze(1).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.model(obs).gather(1, action)
        next_q_values = self.target_model(next_obs).max(1)[0].detach().unsqueeze(1)
        target = reward + self.gamma * (1 - done) * next_q_values

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
