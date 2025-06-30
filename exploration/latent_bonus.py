import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

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
        if len(self.memory) > 1000:  # 可调整 memory 上限
            self.memory.pop(0)

    def fit_latent_space(self):
        if len(self.memory) < 2:
            return  # 防止 PCA 报错

        X = np.array(self.memory)

        # 如果是多时间步数据（[N, T, D]），flatten 成 [N*T, D]
        if X.ndim == 3:
            N, T, D = X.shape
            X = X.reshape(N * T, D)

        Z = self.pca.fit_transform(X)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde.fit(Z)

    def compute_bonus(self, obs):
        if self.kde is None:
            return 0.0

        obs = np.array(obs).reshape(1, -1)
        obs_latent = self.pca.transform(obs)
        log_density = self.kde.score_samples(obs_latent)[0]
        bonus = -self.beta * log_density  # 负对数密度作为探索奖励
        return float(bonus)

