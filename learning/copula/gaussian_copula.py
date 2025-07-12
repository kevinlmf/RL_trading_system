import numpy as np

class GaussianCopula:
    def __init__(self):
        self.fitted = False

    def fit(self, data):
        """
        Fit the Gaussian Copula to the dataset.
        For now, just compute and store empirical mean and covariance.
        """
        print("Fitting Gaussian Copula...")
        self.mean = np.mean(data, axis=0)
        self.cov = np.cov(data, rowvar=False)
        self.fitted = True

    def transform(self, window_data):
        """
        Transform window data into Copula latent space.
        For simplicity, return summary statistics as latent features:
        - Mean and variance for each asset in the window.
        """
        if not self.fitted:
            raise ValueError("GaussianCopula must be fitted before calling transform().")

        mean_features = np.mean(window_data, axis=0)  # shape: (asset_dim,)
        var_features = np.var(window_data, axis=0)    # shape: (asset_dim,)

        # Combine mean and variance into a single latent vector
        latent_features = np.concatenate([mean_features, var_features])  # shape: (asset_dim * 2,)
        return latent_features

