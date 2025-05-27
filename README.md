# Quant_trading_system 🧠📈

A full-featured modular quantitative trading system powered by:

- 🧠 High-performance C++ data loader via PyBind11
- 🤖 Reinforcement Learning with PPO / DQN (Stable-Baselines3)
- 🧩 Custom OpenAI Gym-style trading environment
- 🖥️ Linux-compatible terminal CLI automation (`run_training.sh`)
- 📊 Strategy comparison plots (DQN vs PPO vs Random)
- 🧪 Integrated test framework for random baseline and trained models
- 📦 TensorBoard logging support for training monitoring

---

## 📁 Project Structure

Quant_trading_system/
├── train_dqn.py / train_ppo.py ← DQN/PPO training scripts
├── scripts/ ← CLI scripts
│ ├── run_training.sh ← One-command bash training interface
│ ├── test_model.py ← Load and evaluate trained models
│ ├── compare_strategies.py ← Run & plot DQN vs PPO vs Random
│ └── test_random.py ← Run random baseline on environment
├── env/ ← Trading environment + data loader
│ ├── trading_env.py
│ └── data_loader.py
├── cpp_core/ ← C++ module with PyBind11 binding
│ ├── src/
│ ├── bindings/
│ └── build/ ← Compiled .so lives here
├── models/ ← Saved PPO / DQN models
├── tensorboard/ ← TensorBoard logs
├── data/ ← Input OHLCV CSV (e.g., SPY_1d.csv)
└── README.md


✅ Features Completed
 DQN / PPO RL agents

 Bash CLI training launcher

 Monitor + TensorBoard logs

 C++ module loading test (test_cpp_module.py)

 Strategy evaluation & visualization

 Random baseline test runner

 Linux-based terminal operation

🔭 Future Work
 Add portfolio metrics (Sharpe Ratio, Win Rate, Max Drawdown)

 Integrate order execution module in C++

 Add alpha_engine & risk_control modules

 Implement CSV output for full trading logs

 Hyperparameter tuning via Optuna


