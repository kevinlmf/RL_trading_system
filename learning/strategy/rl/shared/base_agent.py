# 5_learning/strategy/rl/shared/base_agent.py

import os
import abc
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents. 
    All concrete agents (e.g., PPO, DQN) must implement this interface.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config is not None else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def act(self, observation: np.ndarray, evaluation: bool = False) -> int:
        """
        Choose an action given an observation.
        Set `evaluation=True` to disable randomness during inference.
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update agent using a batch of data.
        Return a dictionary of training logs (e.g., loss, KL).
        """
        pass

    def save(self, path: str) -> None:
        """
        Save model parameters to the specified path.
        Requires `self.model` to be defined in the subclass.
        """
        if hasattr(self, "model"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
        else:
            raise NotImplementedError("Subclasses must define self.model to use save().")

    def load(self, path: str) -> None:
        """
        Load model parameters from the specified path.
        Requires `self.model` to be defined in the subclass.
        """
        if hasattr(self, "model"):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            raise NotImplementedError("Subclasses must define self.model to use load().")
