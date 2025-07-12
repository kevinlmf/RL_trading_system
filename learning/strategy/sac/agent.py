import torch
import torch.nn.functional as F
import numpy as np

from .actor import GaussianPolicy
from .critic import QNetwork
from .replay_buffer import ReplayBuffer

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -action_dim

        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(alpha)).to(device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            action, _, _ = self.actor.sample(state, deterministic=True)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return  # Wait until we have enough samples

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_action)
            target_q2 = self.critic2_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = rewards + (1 - dones) * self.gamma * target_q

        # Critic update
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        new_action, log_prob, _ = self.actor.sample(states)
        q1 = self.critic1(states, new_action)
        q2 = self.critic2(states, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (entropy coefficient) update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
