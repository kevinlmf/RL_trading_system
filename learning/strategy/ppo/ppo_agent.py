import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch.distributions import Categorical


class PPOAgent:
    def __init__(self, actor_critic, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                 update_epochs=10, batch_size=64, device="cpu"):

        self.actor_critic = actor_critic
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = device

        self.memory = []

    def store_transition(self, transition):
        self.memory.append(transition)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits, _ = self.actor_critic(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def compute_gae(self, rewards, dones, values, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def train_step(self):
        states, actions, rewards, dones, log_probs, values = zip(*self.memory)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_value = self.actor_critic(states[-1])[1].item()
        advantages, returns = self.compute_gae(rewards, dones, values.tolist(), next_value)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        dataset_size = len(states)
        for _ in range(self.update_epochs):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                logits, value_preds = self.actor_critic(batch_states)
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                value_loss = F.mse_loss(value_preds.squeeze(), batch_returns)

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory = []  # Clear memory after update


