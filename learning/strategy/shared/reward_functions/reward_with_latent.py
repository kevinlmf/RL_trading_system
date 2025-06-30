import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

class SimpleReward:
    def __init__(self):
        self.last_portfolio_value = None

    def reset(self, initial_value):
        self.last_portfolio_value = initial_value

    def __call__(self, current_value):
        if self.last_portfolio_value is None:
            return 0.0
        reward = (current_value - self.last_portfolio_value) / self.last_portfolio_value
        self.last_portfolio_value = current_value
        return reward


class LatentExplorationBonus:
    def __init__(self, n_components=2, beta=0.05, bandwidth=0.2):
        self.n_components = n_components
        self.beta = beta
        self.bandwidth = bandwidth
        self.memory = []
        self.pca = PCA(n_components=n_components)
        self.kde = None

    def update_memory(self, obs):
        self.memory.append(obs)
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def fit_latent_space(self):
        if len(self.memory) < 50:
            return
        X = np.array(self.memory)
        if X.ndim == 3:
            N, T, D = X.shape
            X = X.reshape(N * T, D)
        Z = self.pca.fit_transform(X)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(Z)

    def compute_bonus(self, obs):
        if self.kde is None or len(self.memory) < 50:
            return 0.0
        x = obs.reshape(1, -1)
        z = self.pca.transform(x)
        log_density = self.kde.score_samples(z)[0]
        novelty = -log_density
        return self.beta * novelty


class RewardWithLatentExploration:
    def __init__(self, base_reward: SimpleReward, latent_bonus: LatentExplorationBonus):
        self.base_reward = base_reward
        self.latent_bonus = latent_bonus

    def reset(self, initial_value):
        self.base_reward.reset(initial_value)
        self.latent_bonus.memory = []
        self.latent_bonus.kde = None

    def __call__(self, current_value, obs):
        self.latent_bonus.update_memory(obs)
        self.latent_bonus.fit_latent_space()
        reward = self.base_reward(current_value)
        bonus = self.latent_bonus.compute_bonus(obs)
        return reward + bonus

