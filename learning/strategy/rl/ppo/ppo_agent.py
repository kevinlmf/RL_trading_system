import torch
import torch.nn as nn
import torch.optim as optim
from learning.strategy.rl.ppo.ppo_network import PPOActorCritic
import numpy as np


class PPOAgent:
    def __init__(self, obs_shape, action_dim, gamma=0.99, lam=0.95, lr=3e-4, clip_eps=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        input_dim = obs_shape[0] * obs_shape[1]
        self.model = PPOActorCritic(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, obs):
        obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def select_action(self, obs):
        return self.act(obs)  # ✅ 用于统一接口（和 DQN 接轨）

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, batch):
        obs = torch.tensor([x.flatten() for x in batch["obs"]], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32).to(self.device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32).to(self.device)

        logits, values = self.model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # PPO Clipped Loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(values.squeeze(), returns)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }


