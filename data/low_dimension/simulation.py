import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, rankdata, spearmanr
from scipy.linalg import cholesky
import os

# === Parameters ===
T = 1000
N = 10
nu = 5
np.random.seed(42)

# === Step 1: Simulate synthetic returns Y ===
Y = np.random.randn(T, N)

# === Step 2: ECDF transform to uniform [0,1] ===
U = np.array([rankdata(Y[:, i], method='ordinal') for i in range(N)]).T
U = (U - 0.5) / T

# === Step 3: Inverse t-transform ===
X = t.ppf(U, df=nu)

# === Step 4: Correlation matrix and sample from t-copula ===
base_corr = 0.4
Sigma = np.full((N, N), base_corr)
np.fill_diagonal(Sigma, 1.0)

L = cholesky(Sigma, lower=True)
Z = np.random.standard_t(df=nu, size=(T, N))
X_tilde = Z @ L.T

# === Step 5: Back to copula scale ===
U_tilde = t.cdf(X_tilde, df=nu)

# === Step 6: Save simulated data ===
df = pd.DataFrame(U_tilde, columns=[f"asset_{i+1}" for i in range(N)])
save_path = "3_data/low_dimension/simulated_copula_data.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)
print(f"✅ Saved simulated data to: {save_path}")

# === Step 7: Visualize Marginals ===
plt.figure(figsize=(16, 8))
for i, col in enumerate(df.columns):
    plt.subplot(2, 5, i + 1)
    sns.histplot(df[col], bins=20, kde=False, color='skyblue', edgecolor='black')
    plt.title(col)
    plt.xlim(0, 1)
plt.suptitle("✅ Marginal Distributions (Uniform)", y=1.02)
plt.tight_layout()
plt.show()

# === Step 8: Spearman Correlation Heatmap ===
corr, _ = spearmanr(df)
corr_matrix = pd.DataFrame(corr, index=df.columns, columns=df.columns)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("📊 Spearman Correlation — t-Copula Simulation")
plt.tight_layout()
plt.show()
