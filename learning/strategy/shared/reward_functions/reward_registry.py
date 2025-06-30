from .basic_reward import BasicReward
from .reward_with_latent import RewardWithLatentExploration

REWARD_REGISTRY = {
    "basic": BasicReward,
    "latent": RewardWithLatentExploration,
}
