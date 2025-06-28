import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# === 1. Download Yahoo Finance data ===
tickers = ["SPY", "QQQ", "TLT", "GLD", "BITO"]
print("📥 Downloading data from Yahoo Finance...")
data = yf.download(tickers, start="2022-01-01", end="2024-12-31", auto_adjust=True)

# === 2. Extract 'Close' prices only ===
close_data = data['Close']

# === 3. Compute daily log returns ===
log_returns = np.log(close_data / close_data.shift(1)).dropna()
log_returns.columns = [f"asset_{i+1}" for i in range(len(tickers))]

# === 4. Print and save ===
print("✅ Real-world daily log returns (preview):\n")
print(log_returns.head())

# === 4.5 Save to CSV ===
save_path = "3_data/low_dimension/real_asset_log_returns.csv"
log_returns.to_csv(save_path, index=True)
print(f"✅ Saved log returns to: {save_path}")

# === 5. Spearman correlation matrix ===
def plot_spearman(df):
    corr, _ = spearmanr(df)
    corr_matrix = pd.DataFrame(corr, index=df.columns, columns=df.columns)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("📘 Spearman Correlation — Real Assets")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_spearman(log_returns)





