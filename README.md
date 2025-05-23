# Quant_trading_system

**Modular RL + Risk Control + C++ High-Frequency Trading System**

This repository implements a modular, extensible trading system that combines:
- ✅ Reinforcement Learning (e.g. PPO, DQN)
- ✅ Rule-based baseline strategies
- ✅ Risk control logic (drawdown limits, stop-loss)
- ✅ High-frequency C++ inference using `libtorch`
- ✅ Real-time execution via Python interfaces

---

## 🧠 Project Objective

The goal is to build a production-ready research pipeline for RL-based trading systems that:
- Learns and generates alpha through RL or rule-based logic
- Controls downside risk via robust filters
- Executes orders through real-time inference + broker API
- Integrates cleanly with both Python (training/logic) and C++ (execution)

---

## 📁 Folder Structure Overview

| Path                   | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `alpha_engine/`        | Alpha generation: RL models (`ppo/`, `dqn/`) and `baseline/` strategies |
| └── `inference/`       | Model export and TorchScript inference via `predict.py`                 |
| `risk_control/`        | Risk modules (e.g., `stop_loss.py`, `drawdown_guard.py`)                |
| `cpp_core/`            | C++ inference core (libtorch + pybind11 bindings)                        |
| └── `bindings/`        | Python-C++ interface (`PYBIND11_MODULE`)                                |
| └── `src/`, `include/` | Inference logic and headers                                              |
| `execution_engine/`    | Real-time execution layer (e.g., `api/server.py`, broker adapters)       |
| └── `app/`             | Web or monitor layer (optional)                                         |
| `scripts/`             | CLI scripts: `train_rl.py`, `run_realtime.py`                           |
| `env/`                 | Custom OpenAI Gym environments                                           |
| `config/`              | YAML or JSON hyperparameter configs                                      |
| `data/`                | Sample datasets / preprocessed data                                     |
| `tests/`               | Unit and integration tests                                               |
| `tensorboard_logs/`    | Training visualization logs                                              |
| `.gitignore`           | Ignore cache, logs, `.env`, `.pyc`, and builds                          |
| `Dockerfile`           | Container for portable training/inference                               |
| `requirements.txt`     | Python dependencies                                                      |

---

