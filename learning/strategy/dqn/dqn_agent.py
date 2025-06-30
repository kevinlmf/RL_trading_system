import torch
import torch.nn.functional as F
import numpy as np
from learning.strategy.dqn.replay_buffer import ReplayBuffer
from .dqn_network import QNetwork


class DQNAgent:
    def __init__(self, q_network, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=500,
                 buffer_capacity=10000, batch_size=64):

        self.q_network = q_network
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state):
        self.steps_done += 1
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                        np.exp(-1. * self.steps_done / self.epsilon_decay)

        if np.random.rand() < eps_threshold:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        q_values = self.q_network(state).gather(1, action)
        with torch.no_grad():
            max_next_q = self.target_network(next_state).max(1)[0].unsqueeze(1)
            target = reward + (1 - done) * self.gamma * max_next_q

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


