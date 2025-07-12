import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        print(f"📐 ActorCritic initialized | state_dim={state_dim}, action_dim={action_dim}")

        self.expected_state_dim = state_dim  # ✅ 保存期望的 state_dim

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        # ✅ 确保是 tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # ✅ 确保有 batch 维度
        if state.ndim == 1:
            state = state.unsqueeze(0)  # (state_dim,) ➝ (1, state_dim)

        # ✅ 检查输入维度是否正确
        if state.shape[1] != self.expected_state_dim:
            raise ValueError(
                f"❌ Shape mismatch in ActorCritic.forward: expected {self.expected_state_dim}, "
                f"but got {state.shape[1]}"
            )

        print(f"⚡ ActorCritic.forward | input shape={state.shape}")

        # Forward pass
        x = self.shared(state)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value



