import pandas as pd

def load_csv_data(path):
    # ✅ 跳过前两行说明 + 第三行无用 header
    df = pd.read_csv(path, skiprows=3, header=None)

    # ✅ 手动设置列名
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # ✅ 丢弃 Date 列，只保留 OHLCV 数值
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # ✅ 转换为 float 类型（防止是字符串）
    df = df.astype(float)

    return df.values




