# 🧠 Quant_trading_system 📈

A full-featured modular quantitative trading system powered by:

- 🤖 Reinforcement Learning with PPO / DQN (Stable-Baselines3)
- 🧠 High-performance C++ data loader + execution engine via PyBind11
- 🧩 Custom OpenAI Gym-style trading environment
- 🖥️ Linux/macOS-friendly CLI automation for training & testing
- 📊 Strategy comparison: DQN vs PPO vs Random baseline
- 🧪 Integrated test framework for C++ modules and Python models
- 📦 TensorBoard logging support for training visualization

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your_username/Quant_trading_system.git
cd Quant_trading_system

# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install all required dependencies
pip install -r requirements.txt

# 4. Build the C++ PyBind11 backend module
mkdir -p cpp_core/build && cd cpp_core/build
cmake ..
make -j4
cd ../..
```

---

## ✅ Run the System

```bash
# Run C++ backend sanity check
python scripts/test_cpp_module.py

# Run random baseline strategy
python scripts/test_random.py

# Train PPO agent
python train_ppo.py

# Train DQN agent
python train_dqn.py

# Visualize and compare strategies
python scripts/compare_strategies.py
```

---

## 📁 Project Structure

```
Quant_trading_system/
├── train_dqn.py / train_ppo.py         ← RL training entrypoints
├── scripts/                            ← CLI tools
│   ├── run_training.sh                 ← One-command training runner
│   ├── compare_strategies.py           ← Visualize results
│   ├── test_model.py                   ← Evaluate saved models
│   ├── test_random.py                  ← Random baseline
│   └── test_cpp_module.py              ← C++ module sanity test
├── env/                                ← Gym-style trading environment
│   ├── trading_env.py
│   └── data_loader.py
├── cpp_core/                           ← C++ backend with PyBind11
│   ├── include/                        ← Header files
│   ├── src/                            ← C++ implementations
│   ├── bindings/                       ← PyBind11 wrappers
│   ├── build/                          ← Output for `cpp_trading.so`
│   └── CMakeLists.txt                  ← Build instructions
├── models/                             ← Saved models
├── tensorboard/                        ← Training logs
├── data/                               ← OHLCV data (e.g., SPY_1d.csv)
└── README.md                           ← You're here!
```

---

## ✅ Features Completed

- ✅ PPO / DQN reinforcement learning agents
- ✅ Random baseline strategy
- ✅ Modular `gym.Env` trading environment
- ✅ PyBind11 C++ backend (data feed + order execution)
- ✅ One-line bash training launcher
- ✅ Strategy comparison plotting
- ✅ C++ module test runner
- ✅ TensorBoard logging

---

## 🛠️ Future Work

- 📈 Add portfolio metrics (Sharpe Ratio, Win Rate, Max Drawdown)
- ⚙️ Real-time order execution in C++
- 🧠 `alpha_engine` and `risk_control` strategy modules
- 📁 Export full trade logs to CSV
- 🔍 Hyperparameter tuning via Optuna

---

## 📦 Requirements

Dependencies (see `requirements.txt`):

```
# RL & Environment
stable-baselines3==1.8.0
gymnasium==0.29.1

# Scientific Stack
numpy>=1.23
pandas>=1.5
matplotlib>=3.6

# PyBind11 Module
pybind11>=2.11

# Logging
tensorboard>=2.10

# Optional
scikit-learn>=1.2
```

---

## 📄 License

MIT License © 2025 Mengfan Long

