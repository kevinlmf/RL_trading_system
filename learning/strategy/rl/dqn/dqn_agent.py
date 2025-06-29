import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_shape, n_actions, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3):
        self.model = QNetwork(obs_shape, n_actions)
        self.target_model = QNetwork(obs_shape, n_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criteria = nn.MSELoss()
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_actions = n_actions

    def select_action(self, obs, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        return q_values.argmax().item()

    def remember(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s_, done = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        s_ = torch.tensor(np.array(s_), dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        q_values = self.model(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_model(s_).max(1)[0].unsqueeze(1)
            target = r + self.gamma * next_q * (1 - done)

        loss = self.criteria(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

