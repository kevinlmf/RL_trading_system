from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import numpy as np

class LatentFactorBonus:
    def __init__(self, n_components=2, beta=0.1, bandwidth=0.2):
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
        X = np.array(self.memory)
        Z = self.pca.fit_transform(X)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde.fit(Z)

    def compute_bonus(self, obs):
        if len(self.memory) < 50:
            return 0.0
        z = self.pca.transform(obs.reshape(1, -1))
        log_density = self.kde.score_samples(z)
        bonus = -self.beta * log_density[0]
        return bonus
