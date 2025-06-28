
import torch.nn as nn
import numpy as np

class DQNNet(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        input_dim = np.prod(obs_shape)  # Flatten (e.g., 30×10 => 300)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten each obs in batch
        return self.net(x)

