import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, rankdata
from scipy.linalg import cholesky
import os

# === 模拟参数 ===
T = 1000           # 时间长度
N = 33             # 资产数量
nu = 5             # 自由度 (t-copula)
base_corr = 0.4    # 基础相关性
np.random.seed(42)

# === Step 0: 加载真实资产收益率，用于经验分布 ===
real_data_path = "data/mid_dimension/real_asset_log_returns.csv"
real_df = pd.read_csv(real_data_path)

# 去除日期列并限制列数
if "Date" in real_df.columns:
    real_df = real_df.drop(columns=["Date"])
real_df = real_df.iloc[:, :N]

print(f"📥 Loaded real asset returns shape: {real_df.shape}")

# === Step 1: 初始化标准正态变量并转为 copula 样本 ===
Y = np.random.randn(T, N)
U = np.array([rankdata(Y[:, i], method='ordinal') for i in range(N)]).T
U = (U - 0.5) / T  # 将 ranks 转为 (0,1)
X = t.ppf(U, df=nu)

# === Step 2: 构建协方差结构并采样 t-copula ===
Sigma = np.full((N, N), base_corr)
np.fill_diagonal(Sigma, 1.0)
L = cholesky(Sigma, lower=True)

Z = np.random.standard_t(df=nu, size=(T, N))
X_tilde = Z @ L.T
U_tilde = t.cdf(X_tilde, df=nu)

# === Step 3: Copula 样本 → 原始收益率（经验反推） ===
def inverse_empirical_cdf(u_values, real_samples):
    sorted_samples = np.sort(real_samples)
    n = len(sorted_samples)
    idx = (u_values * n).astype(int).clip(0, n - 1)
    return sorted_samples[idx]

Y_tilde = np.zeros_like(U_tilde)
for i in range(N):
    Y_tilde[:, i] = inverse_empirical_cdf(U_tilde[:, i], real_df.iloc[:, i].values)

# === Step 4: 保存模拟数据 ===
copula_df = pd.DataFrame(Y_tilde, columns=[f"asset_{i+1}" for i in range(N)])
save_path = "data/mid_dimension/simulated_copula_returns.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
copula_df.to_csv(save_path, index=False)

print(f"✅ Simulated t-Copula Returns Saved to: {save_path}")
print("📈 Sample mean/std:\n", copula_df.describe().loc[["mean", "std"]])

# === Step 5: 可视化单个资产的分布 ===
plt.figure(figsize=(8, 4))
sns.histplot(copula_df['asset_1'], bins=20, kde=True)
plt.title("Simulated Asset 1 Return Distribution")
plt.xlabel("Return")
plt.tight_layout()
plt.show()
