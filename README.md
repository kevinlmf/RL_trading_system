# 🧠 Quant_trading_system 📈

A full-featured modular quantitative trading system powered by:

- 🤖 Reinforcement Learning with PPO / DQN (Stable-Baselines3)
- 🧠 High-performance C++ data loader + execution module via PyBind11
- 🧩 Custom OpenAI Gym-style trading environment
- 🖥️ Linux-friendly CLI automation for training & testing
- 📊 Strategy comparison (DQN vs PPO vs Random baseline)
- 🧪 Integrated test framework for C++ modules and Python models
- 📦 TensorBoard logging support for live training monitoring

---

## 📁 Project Structure

```text
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

✅ TensorBoard log support

🐧 Fully compatible with Linux, WSL2, and macOS


🛠️ Future Work
📈 Add portfolio metrics (Sharpe Ratio, Win Rate, Max Drawdown)

⚙️ Integrate real-time order execution module in C++

🧠 Add alpha_engine and risk_control strategy modules

📁 Export full trade logs as CSV

🔍 Hyperparameter tuning via Optuna

