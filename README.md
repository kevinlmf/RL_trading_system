# 🧠 RL_trading_system

A reinforcement learning-based trading system integrating Random, Buy-and-Hold, DQN, and PPO strategies on simulated and real market data. Built with modular Python code and clean environment interfaces.

---

## 📂 Project Structure

RL_trading_system/
├── 2_theory/ # Theoretical background (e.g. RL, Finance, Copula)
├── 3_data/ # Raw and processed data (real & simulated)
├── 4_learning/
│ ├── env/ # TradingEnv: gym-compatible market simulator
│ ├── strategy/
│ │ ├── random_strategy.py
│ │ ├── buy_and_hold_strategy.py
│ │ ├── dqn/
│ │ │ ├── dqn_agent.py
│ │ │ └── dqn_network.py
│ │ └── ppo/
│ │ ├── ppo_agent.py
│ │ └── ppo_network.py
├── 5_evaluation/
│ └── evaluate_strategies.py
├── models/ # Trained models (e.g., ppo_actor_critic.pt, dqn_model.pt)
├── scripts/ # Training scripts (e.g., train_dqn.py, train_ppo.py)
├── README.md
└── requirements.txt

---

## 🚀 Features

- ✅ **Trading Environment**: Supports variable window size, multi-asset simulation, cash dynamics.
- ✅ **Strategies**:
  - `RandomStrategy`: Uniform random action
  - `BuyAndHoldStrategy`: Static full-investment baseline
  - `DQN`: Deep Q-Learning with target network and replay buffer
  - `PPO`: Actor-Critic with clipped policy gradient
- ✅ **Evaluation**:
  - Standardized metrics:
    - Total Reward
    - Sharpe Ratio
    - Max Drawdown
  - One-line performance comparison plot
- ✅ **Modular & Extendable**: Easy to add more strategies, reward functions, and environments.

---

## 🏋️ Training

### 🟦 PPO

```bash
python scripts/train_ppo.py
Trains PPOAgent using TradingEnv, saves to models/ppo_actor_critic.pt.

🟩 DQN
```bash
python scripts/train_dqn.py
Trains DQNAgent using experience replay and target network. Saves model to models/dqn_model.pt.

📊 Evaluation
```bash
python evaluation/evaluate_strategies.py
Outputs performance metrics for all strategies and saves a plot at:

```bash
evaluation/portfolio_comparison_full.png
Example Output:

```yaml
📊 Random Strategy:
{'Total Reward': 2629000.71, 'Sharpe Ratio': 0.007, 'Max Drawdown': 7913689.80}

📊 Buy and Hold Strategy:
{'Total Reward': 3124202.82, 'Sharpe Ratio': 0.009, 'Max Drawdown': 8076941.50}

📊 PPO Strategy:
{'Total Reward': 4215084.50, 'Sharpe Ratio': 0.012, 'Max Drawdown': 8027697.94}

📊 DQN Strategy:
{'Total Reward': 280XXXX.XX, 'Sharpe Ratio': 0.XXX, 'Max Drawdown': XXXXXXXX.XX}
📌 Dependencies
Install from requirements.txt:

```bash
pip install -r requirements.txt
Python 3.10+ recommended.

🔐 Model Checkpoints
All trained models are saved in the models/ folder:

ppo_actor_critic.pt

dqn_model.pt

If missing, retrain using train_ppo.py or train_dqn.py.

💡 TODO
 Add Copula-based reward shaping

 Add MBIE exploration bonus

 Improve portfolio risk metrics (e.g., Sortino Ratio, Calmar Ratio)

 Backtest on real Yahoo Finance log returns

 Add Stable-Baselines3 compatibility

