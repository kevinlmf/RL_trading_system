import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CPPOAgent:
    def __init__(self, actor_critic, lr=3e-4, gamma=0.99, clip_epsilon=0.2, cvar_alpha=0.05,
                 cvar_weight=1.0, lagrange_lr=0.01, batch_size=64, device="cpu"):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight
        self.lagrange_lr = lagrange_lr
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Lagrange multiplier for CVaR constraint
        self.lagrange_multiplier = torch.tensor(1.0, requires_grad=True, device=self.device)

        # Buffers
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.buffer_log_probs = []
        self.buffer_values = []

    def select_action(self, state):
        """Select action based on current policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.actor_critic(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item()

    def store_transition(self, transition):
        state, action, reward, done, log_prob, value = transition
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)

    def train_step(self):
        """Update policy with CPPO objective."""
        # === Convert buffers ===
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

        # === CVaR penalty ===
        sorted_returns, _ = torch.sort(returns)
        var_threshold = int(self.cvar_alpha * len(sorted_returns))
        cvar_value = sorted_returns[:var_threshold].mean()

        # === PPO update with CVaR constraint ===
        for _ in range(4):  # Multiple epochs
            logits, state_values = self.actor_critic(states)
            action_dist = torch.distributions.Categorical(logits)
            log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()

            # Ratio for clipped objective
            ratios = torch.exp(log_probs - old_log_probs)
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # Value loss
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)

            # Lagrangian term for CVaR
            cvar_penalty = self.lagrange_multiplier * (self.cvar_weight * cvar_value)

            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss + cvar_penalty - 0.01 * entropy

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
            self.optimizer.step()

        # === Update Lagrange multiplier ===
        with torch.no_grad():
            self.lagrange_multiplier += self.lagrange_lr * (cvar_value - 0)  # Target CVaR=0
            self.lagrange_multiplier = torch.clamp(self.lagrange_multiplier, min=0.0)

        self.clear_buffer()

    def clear_buffer(self):
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.buffer_log_probs = []
        self.buffer_values = []
