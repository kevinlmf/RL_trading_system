import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),  # ✅ 修复变量名
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # 🧩 保证 batch dim
        x = x.view(x.size(0), -1)  # 🧩 Flatten
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared).squeeze(-1)
        return logits, value

    def get_action(self, x, deterministic=False):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
