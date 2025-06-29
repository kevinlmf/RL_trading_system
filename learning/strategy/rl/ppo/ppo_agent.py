import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, obs_shape, n_actions, clip_eps=0.2, gamma=0.99, lr=3e-4):
        self.model = ActorCritic(obs_shape, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.n_actions = n_actions

    def select_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.model(obs_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()

    def learn(self, trajectory):
        obs_list, action_list, logprob_list, reward_list, value_list = zip(*trajectory)

        # Compute returns and advantages
        returns, advantages = [], []
        G = 0
        for t in reversed(range(len(reward_list))):
            G = reward_list[t] + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(value_list, dtype=torch.float32)
        advantages = returns - values.detach()

        # Convert to tensor
        obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32)
        action_batch = torch.tensor(action_list)
        old_log_probs = torch.stack(logprob_list)

        logits, value_preds = self.model(obs_batch)
        dist = torch.distributions.Categorical(logits.softmax(dim=-1))
        new_log_probs = dist.log_prob(action_batch)

        ratio = (new_log_probs - old_log_probs).exp()
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        critic_loss = (returns - value_preds.squeeze()).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


