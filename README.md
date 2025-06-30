# 📈 RL Trading System

A modular reinforcement learning (RL) system for financial trading, integrating:
- 🧠 PPO / DQN strategies
- 📊 t-Copula-based synthetic data simulation
- 🧮 Risk-aware exploration via latent density bonus
- ⚙️ Modular Python + PyBind11 C++ backend

---

## 🔧 Project Structure

```
RL_trading_system/
├── Dockerfile                       # Docker container setup (optional)
├── README.md                        # Project overview (README)
├── cpp_implementation               # C++ modules (for high performance)
├── data                              # Raw & simulated data (asset returns, simulations)
├── evaluation                        # Strategy evaluation scripts
├── exploration                       # Latent bonus exploration module
├── learning                          # RL agents, envs, reward functions
├── models                            # Saved model checkpoints (.pt files)
├── requirements.txt                  # Python dependencies
├── scripts                           # Training and testing entry points
├── theory                            # RL theory and copula modeling
└── venv                              # Virtual environment
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone git@github.com:kevinlmf/RL_trading_system.git
cd RL_trading_system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Download real asset returns
python 3_data/mid_dimension/download_real_data.py

# Simulate synthetic data via t-Copula
python 3_data/mid_dimension/simulate_copula_data.py
```

---

## 🧠 Training

### Train PPO Agent

```bash
python scripts/train_ppo.py
```

### Train DQN Agent

```bash
python scripts/train_dqn.py
```

---

## 📈 Evaluation

```bash
python evaluation/evaluate_strategies.py
```

This will output:
- 💹 Portfolio performance plot
- 🧮 Sharpe ratio, max drawdown, total return
- 🔁 Comparison across PPO / DQN / Buy-and-Hold / Random

---

## 🧪 Exploration Bonus

This system includes an exploration bonus, using PCA and Kernel Density Estimation (KDE) to reward exploration in the latent space.

---

## 🧪 Requirements

```txt
# RL & Env
stable-baselines3==2.2.1
gymnasium==0.29.1
gym==0.26.2              # Legacy Gym API compatibility

# Scientific stack
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
scipy>=1.8               # t-distribution, Cholesky decomposition
seaborn>=0.12            # Correlation heatmaps and advanced plots

# C++ module (optional)
pybind11>=2.11

# Logging
tensorboard>=2.10

# Optional
scikit-learn>=1.2
joblib>=1.2              # Used in scikit-learn and stable-baselines3
```

---



## 📌 Future Extensions

- [ ] Copula-informed reward shaping
- [ ] MBIE-EB style optimistic exploration
- [ ] Offline RL & Behavior Cloning
- [ ] Deep hedging module with stochastic pricing
