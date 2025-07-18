import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # === Shared layers ===
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # === Actor network ===
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Probability distribution over actions
        )

        # === Critic network ===
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # State value
        )

    def forward(self, state):
        """
        Forward pass through shared layers, actor, and critic.
        Handles both single and batched states.
        """
        # Ensure state is a torch tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Add batch dimension if needed
        if state.ndim == 1:
            state = state.unsqueeze(0)  # (state_dim,) -> (1, state_dim)

        x = self.shared(state)
        action_probs = self.actor(x)
        state_value = self.critic(x).squeeze(-1)  # (batch_size, 1) -> (batch_size,)

        return action_probs, state_value

