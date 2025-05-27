# 🧠 Quant_trading_system 📈

A full-featured modular quantitative trading system powered by:

- 🤖 Reinforcement Learning with PPO / DQN (Stable-Baselines3)
- 🧠 High-performance C++ data loader + execution module via PyBind11
- 🧩 Custom OpenAI Gym-style trading environment
- 🖥️ Linux/macOS-friendly CLI automation for training & testing
- 📊 Strategy comparison (DQN vs PPO vs Random baseline)
- 🧪 Integrated test framework for C++ modules and Python models
- 📦 TensorBoard logging support for live training monitoring

---

## 🚀 Quick Start

Clone the project and set up your environment in **4 simple steps**:

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

Once setup is done, test the system:
# Run C++ backend sanity check
python scripts/test_cpp_module.py

# Run random baseline strategy
python scripts/test_random.py

Train a reinforcement learning agent:
# Train PPO agent
python train_ppo.py

# Train DQN agent
python train_dqn.py

# Visualize and compare strategies
python scripts/compare_strategies.py


📁 Project Structure
Quant_trading_system/
├── train_dqn.py / train_ppo.py         ← RL training entrypoints
├── scripts/                            ← CLI tools
│   ├── run_training.sh                 ← One-command training runner
│   ├── compare_strategies.py           ← Visualize DQN / PPO / Random
│   ├── test_model.py                   ← Evaluate saved models
│   ├── test_random.py                  ← Run random baseline
│   └── test_cpp_module.py              ← Sanity test for cpp_trading.so
├── env/                                ← Gym-style trading environment
│   ├── trading_env.py
│   └── data_loader.py
├── cpp_core/                           ← 🧩 C++ backend with PyBind11
│   ├── include/                        ← C++ Header files (interfaces)
│   │   ├── data_feed.h                 ← DataFeed for OHLCV
│   │   └── order_executor.hpp          ← Mock order execution logic
│   ├── src/                            ← C++ implementations
│   │   ├── data_feed.cpp
│   │   └── order_executor.cpp
│   ├── bindings/                       ← Python-C++ interface via PyBind11
│   │   ├── data_bindings.cpp
│   │   ├── order_bindings.cpp
│   │   └── main_bindings.cpp           ← PYBIND11_MODULE entry
│   ├── build/                          ← Output directory for `cpp_trading.so`
│   └── CMakeLists.txt                  ← Build instructions using pybind11_add_module
├── models/                             ← Saved RL agent models (PPO / DQN)
├── tensorboard/                        ← Training logs for visualization
├── data/                               ← OHLCV data files (e.g., `SPY_1d.csv`)
└── README.md                           ← You're here!

✅ Features Completed
✅ PPO / DQN reinforcement learning agents

✅ Random baseline strategy

✅ Modular training environment using gym.Env

✅ C++ module integration with PyBind11 (data feed + order execution)

✅ One-line bash training launcher (run_training.sh)

✅ Strategy comparison plotting (matplotlib + CSV evaluation)

✅ C++ module test runner

✅ TensorBoard logging support

🛠️ Future Work
📈 Add portfolio metrics (Sharpe Ratio, Win Rate, Max Drawdown)

⚙️ Integrate real-time order execution module in C++

🧠 Add alpha_engine and risk_control strategy modules

📁 Export full trade logs as CSV

🔍 Hyperparameter tuning via Optuna


🧪 Requirements
See requirements.txt, which includes:
# RL & environment
stable-baselines3==1.8.0
gymnasium==0.29.1

# Classic scientific stack
numpy>=1.23
pandas>=1.5
matplotlib>=3.6

# C++ module (PyBind11)
pybind11>=2.11

# Logging
tensorboard>=2.10

# Optional
scikit-learn>=1.2

📄 License
MIT License © 2025 Mengfan Long

