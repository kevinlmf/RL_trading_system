import numpy as np

class TFactorCopula:
    def __init__(self, latent_dim=3, lambda_weight=0.5, max_iter=300):
        self.latent_dim = latent_dim  # ✅ 固定 latent_dim
        self.lambda_weight = lambda_weight
        self.max_iter = max_iter
        self.Lambda = None
        self.Psi = None
        self.nu = 10.0  # 固定自由度，防止奇异

    def fit(self, data):
        T, N = data.shape
        K = self.latent_dim
        np.random.seed(42)

        print(f"🔄 Fitting t-Factor Copula | fixed latent_dim={K}")

        # === 初始化 ===
        Lambda = np.random.normal(0, 0.1, size=(N, K))
        Psi = np.var(data, axis=0)

        for iter in range(self.max_iter):
            Sigma_inv = np.linalg.inv(Lambda @ Lambda.T + np.diag(Psi))
            tau_inv = np.zeros(T)
            Ez = np.zeros((T, K))
            Ezz = np.zeros((K, K))

            for t in range(T):
                x_t = data[t].reshape(-1, 1)
                delta_t = (x_t.T @ Sigma_inv @ x_t).item()
                shape = (self.nu + N) / 2
                scale = (self.nu + delta_t) / 2
                tau_inv[t] = scale / (shape - 1)

                temp = np.linalg.inv(Lambda.T @ Sigma_inv @ Lambda + np.eye(K))
                M_t = temp @ Lambda.T @ Sigma_inv * np.sqrt(tau_inv[t])
                Ez[t] = (M_t @ x_t).flatten()
                Ezz += tau_inv[t] * (temp + np.outer(Ez[t], Ez[t]))

            # === 更新 Lambda 和 Psi ===
            numerator = np.zeros((N, K))
            denominator = np.zeros((K, K))
            for t in range(T):
                numerator += tau_inv[t] * np.outer(data[t], Ez[t])
                denominator += tau_inv[t] * (np.outer(Ez[t], Ez[t]) + np.eye(K))
            Lambda_new = numerator @ np.linalg.inv(denominator)

            residual = data - Ez @ Lambda_new.T
            Psi_new = np.mean((residual**2).T * tau_inv, axis=1)

            delta = np.linalg.norm(Lambda - Lambda_new) + np.linalg.norm(Psi - Psi_new)
            Lambda, Psi = Lambda_new, Psi_new
            if delta < 1e-5:
                print(f"✅ EM converged at iteration {iter}")
                break

        self.Lambda = Lambda
        self.Psi = Psi
        print(f"✅ t-Factor Copula fitted | latent_dim={K}")

    def transform(self, data):
        if self.Lambda is None:
            raise ValueError("❌ Copula model not fitted.")
        Sigma_inv = np.linalg.inv(self.Lambda @ self.Lambda.T + np.diag(self.Psi))
        T, K = data.shape[0], self.Lambda.shape[1]
        latent_Z = np.zeros((T, K))
        for t in range(T):
            x_t = data[t].reshape(-1, 1)
            temp = np.linalg.inv(self.Lambda.T @ Sigma_inv @ self.Lambda + np.eye(K))
            M_t = temp @ self.Lambda.T @ Sigma_inv
            latent_Z[t] = (M_t @ x_t).flatten()
        return latent_Z
