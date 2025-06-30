import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

class LatentFactorBonus:
    """
    Computes an exploration bonus based on the density in a reduced latent space.
    Uses PCA for dimensionality reduction and Kernel Density Estimation (KDE)
    to assign low-density regions a higher exploration bonus.
    """

    def __init__(self, n_components=2, beta=0.1, bandwidth=0.2):
        """
        Parameters:
            n_components (int): Number of latent dimensions for PCA.
            beta (float): Scaling factor for the exploration bonus.
            bandwidth (float): Bandwidth for the KDE.
        """
        self.n_components = n_components
        self.beta = beta
        self.bandwidth = bandwidth
        self.memory = []         # Stores observed states for density estimation
        self.pca = PCA(n_components=n_components)
        self.kde = None          # KDE will be fit later on latent space

    def update_memory(self, obs):
        """
        Add new observation to memory (FIFO queue with max size 1000).
        """
        self.memory.append(obs)
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def fit_latent_space(self):
        """
        Fit PCA and KDE models on the current memory.
        """
        if len(self.memory) < 2:
            return  # Prevent PCA error when data is insufficient

        X = np.array(self.memory)

        # Flatten 3D sequence data [N, T, D] into [N*T, D] if needed
        if X.ndim == 3:
            N, T, D = X.shape
            X = X.reshape(N * T, D)

        Z = self.pca.fit_transform(X)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde.fit(Z)

    def compute_bonus(self, obs):
        """
        Compute exploration bonus for a given observation based on log-density.
        Returns:
            float: Negative log-probability scaled by beta (encourages low-density regions)
        """
        if self.kde is None:
            return 0.0

        obs = np.array(obs).reshape(1, -1)
        obs_latent = self.pca.transform(obs)
        log_density = self.kde.score_samples(obs_latent)[0]
        bonus = -self.beta * log_density
        return float(bonus)

