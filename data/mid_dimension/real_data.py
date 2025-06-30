import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

# === 1. Define a diverse set of 30+ financial assets ===
tickers = [
    "SPY", "QQQ", "DIA", "IWM", "VTI",               # US Equity Indices
    "TLT", "IEF", "SHY", "BND", "LQD",               # Bonds / Fixed Income
    "GLD", "SLV", "USO", "UNG", "DBC",               # Commodities
    "BITO", "GBTC", "ETHE",                          # Crypto-Linked ETFs
    "MTUM", "VLUE", "QUAL", "USMV", "SIZE",          # Factor ETFs
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLI",         # Sector ETFs (SPDR)
    "XLU", "XLB", "XLRE", "XLC"                      # More sectors (SPDR)
]

# === 2. Download historical price data from Yahoo Finance ===
print("📥 Downloading data from Yahoo Finance...")
data = yf.download(tickers, start="2022-01-01", end="2024-12-31", auto_adjust=True)
close_data = data['Close']

# === 3. Compute daily log returns from closing prices ===
log_returns = np.log(close_data / close_data.shift(1)).dropna()
log_returns.columns = [f"asset_{i+1}" for i in range(len(tickers))]

print("✅ Real-world daily log returns (preview):")
print(log_returns.head())

# === 4. Save the log returns to CSV for later use ===
save_path = "data/mid_dimension/real_asset_log_returns.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
log_returns.to_csv(save_path, index=True)
print(f"✅ Saved log returns to: {save_path}")

# === 5. Visualize the Spearman rank correlation matrix ===
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
