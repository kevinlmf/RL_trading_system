import numpy as np

def compute_metrics(portfolio_values):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    cumulative_return = portfolio_values[-1] / portfolio_values[0] - 1
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    max_drawdown = compute_max_drawdown(portfolio_values)
    return {
        "Cumulative Return": cumulative_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

def compute_max_drawdown(values):
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return np.max(drawdown)
