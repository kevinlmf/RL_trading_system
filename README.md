# RL_trading_system

This project implements a modular reinforcement learning (RL) trading system with multiple strategies including PPO, DQN, and Random policies. It features a custom trading environment and supports evaluation on both simulated (Copula-based) and real financial data.

---

## 📁 Project Structure

```
RL_trading_system/
│
├── scripts/              # Training, testing, and data download scripts
│
├── cpp_implementation/   # (Optional) High-performance C++ modules (if used)
│
├── theory/               # LaTeX documents: financial math, RL theory, copula modeling
│
├── 3_data/
│   ├── low_dimension/      # Simulated or real market data
│   └── processed/          # Cleaned CSVs
│
├── 4_learning/
│   ├── env/                # Custom TradingEnv implementation
│   └── strategy/
│       ├── rl/
│       │   ├── dqn/        # DQN agent and model
│       │   ├── ppo/        # PPO agent and model
│       │   └── random/     # Random policy baseline
│       └── shared/         # Common reward functions, utilities, etc.
│
├── 5_evaluation/           # Scripts for evaluating and comparing strategies
│
├── Dockerfile              # Environment setup (optional)
├── requirements.txt        # Python dependencies
└── README.md               # Project introduction (this file)
```

---

## 🧠 Key Features

- **Custom Trading Environment:** With position tracking, account balance, and action history
- **Multiple Strategies:** PPO, DQN, and a Random policy baseline
- **Copula-Simulated Market Data:** For structured risk modeling
- **Evaluation Metrics:** Total reward, Sharpe ratio, maximum drawdown
- **Modular Design:** Clean separation of strategy, environment, and evaluation

---

## 🚀 Getting Started

### 1. Clone this repository
```bash
git clone https://github.com/kevinlmf/RL_trading_system.git
cd RL_trading_system
```

### 2. Set up the environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run training (e.g., PPO)
```bash
python 0_scripts/train_ppo.py
```

### 4. Evaluate strategies
```bash
python 5_evaluation/evaluate_strategies.py
```

---

## 🧩 Notes

- PPO uses Generalized Advantage Estimation (GAE)
- DQN uses epsilon-greedy exploration with replay buffer
- Trading environment supports discrete Buy / Hold / Sell actions
- Evaluation averages over multiple episodes

---

## 📬 Contact

Feel free to reach out via [GitHub Issues](https://github.com/kevinlmf/RL_trading_system/issues) or [LinkedIn](https://www.linkedin.com/in/yourprofile/).
