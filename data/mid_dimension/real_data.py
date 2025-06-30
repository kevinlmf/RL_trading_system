import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

# === 1. Define 30 diverse assets ===
tickers = [
    "SPY", "QQQ", "DIA", "IWM", "VTI",               # 指数
    "TLT", "IEF", "SHY", "BND", "LQD",               # 债券
    "GLD", "SLV", "USO", "UNG", "DBC",               # 商品
    "BITO", "GBTC", "ETHE",                          # 加密ETF（模拟）
    "MTUM", "VLUE", "QUAL", "USMV", "SIZE",          # 风格因子
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLI", 
    "XLU", "XLB", "XLRE", "XLC"                      # 行业ETF（GICS 11大类）
]

# === 2. Download price data ===
print("📥 Downloading data from Yahoo Finance...")
data = yf.download(tickers, start="2022-01-01", end="2024-12-31", auto_adjust=True)
close_data = data['Close']

# === 3. Compute log returns ===
log_returns = np.log(close_data / close_data.shift(1)).dropna()
log_returns.columns = [f"asset_{i+1}" for i in range(len(tickers))]

print("✅ Real-world daily log returns (preview):")
print(log_returns.head())

# === 4. Save log return data ===
save_path = "data/mid_dimension/real_asset_log_returns.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
log_returns.to_csv(save_path, index=True)
print(f"✅ Saved log returns to: {save_path}")

# === 5. Plot Spearman correlation matrix ===
def plot_spearman(df):
    corr, _ = spearmanr(df)
    corr_matrix = pd.DataFrame(corr, index=df.columns, columns=df.columns)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", square=True)
    plt.title("📘 Spearman Correlation — Real Assets (30-Dim)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_spearman(log_returns)
