import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOAgent:
    def __init__(self, actor_critic, lr=3e-4, gamma=0.99, clip_epsilon=0.2, batch_size=64, device="cpu"):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Buffers
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.buffer_log_probs = []
        self.buffer_values = []

    def select_action(self, state):
        """Choose action based on current policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dim
        with torch.no_grad():
            action_probs, value = self.actor_critic(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item()

    def store_transition(self, transition):
        """Store transition (state, action, reward, done, log_prob, value)."""
        state, action, reward, done, log_prob, value = transition
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)

    def train_step(self):
        """Update policy and value network using PPO loss."""
        # === Efficient buffer conversion ===
        states = torch.tensor(np.array(self.buffer_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(self.buffer_actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(self.buffer_rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(self.buffer_dones), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(self.buffer_log_probs), dtype=torch.float32).to(self.device)
        values = torch.tensor(np.array(self.buffer_values), dtype=torch.float32).to(self.device)

        # === Compute returns and advantages ===
        returns = []
        G = 0
        for reward, done_flag in zip(reversed(rewards), reversed(dones)):
            if done_flag:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - values

        # === PPO Policy Update ===
        for _ in range(4):  # Multiple epochs
            logits, state_values = self.actor_critic(states)
            action_dist = torch.distributions.Categorical(logits)
            log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()

            # Ratio for clipped objective
            ratios = torch.exp(log_probs - old_log_probs)

            # PPO clipped loss
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # Value function loss
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)  # Gradient clipping
            self.optimizer.step()

        # Clear buffers after update
        self.clear_buffer()

    def clear_buffer(self):
        """Clear all stored transitions."""
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.buffer_log_probs = []
        self.buffer_values = []



