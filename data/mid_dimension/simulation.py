import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, rankdata
from scipy.linalg import cholesky
import os

# === Simulation Parameters ===
T = 1000           # Number of time steps (observations)
N = 33             # Number of assets (dimensions)
nu = 5             # Degrees of freedom for t-copula
base_corr = 0.4    # Base off-diagonal correlation
np.random.seed(42) # For reproducibility

# === Step 0: Load real asset returns to build empirical CDFs ===
real_data_path = "data/mid_dimension/real_asset_log_returns.csv"
real_df = pd.read_csv(real_data_path)

# Drop Date column if it exists and select first N assets
if "Date" in real_df.columns:
    real_df = real_df.drop(columns=["Date"])
real_df = real_df.iloc[:, :N]

print(f"📥 Loaded real asset returns shape: {real_df.shape}")

# === Step 1: Generate i.i.d. standard normal samples and transform to uniform (copula) ===
Y = np.random.randn(T, N)
U = np.array([rankdata(Y[:, i], method='ordinal') for i in range(N)]).T
U = (U - 0.5) / T  # Normalize ranks to (0, 1)
X = t.ppf(U, df=nu)  # Convert uniform to t-distribution (marginal)

# === Step 2: Build correlation matrix and generate correlated t-copula samples ===
Sigma = np.full((N, N), base_corr)
np.fill_diagonal(Sigma, 1.0)
L = cholesky(Sigma, lower=True)

Z = np.random.standard_t(df=nu, size=(T, N))  # Uncorrelated t samples
X_tilde = Z @ L.T                             # Apply Cholesky to induce correlation
U_tilde = t.cdf(X_tilde, df=nu)               # Convert back to uniform via CDF

# === Step 3: Map uniform values to real returns using empirical inverse CDF ===
def inverse_empirical_cdf(u_values, real_samples):
    sorted_samples = np.sort(real_samples)
    n = len(sorted_samples)
    idx = (u_values * n).astype(int).clip(0, n - 1)
    return sorted_samples[idx]

Y_tilde = np.zeros_like(U_tilde)
for i in range(N):
    Y_tilde[:, i] = inverse_empirical_cdf(U_tilde[:, i], real_df.iloc[:, i].values)

# === Step 4: Save the simulated returns to CSV ===
copula_df = pd.DataFrame(Y_tilde, columns=[f"asset_{i+1}" for i in range(N)])
save_path = "data/mid_dimension/simulated_copula_returns.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
copula_df.to_csv(save_path, index=False)

print(f"✅ Simulated t-Copula Returns Saved to: {save_path}")
print("📈 Sample mean/std:\n", copula_df.describe().loc[["mean", "std"]])

# === Step 5: Visualize return distribution of the first asset ===
plt.figure(figsize=(8, 4))
sns.histplot(copula_df['asset_1'], bins=20, kde=True)
plt.title("Simulated Asset 1 Return Distribution")
plt.xlabel("Return")
plt.tight_layout()
plt.show()

