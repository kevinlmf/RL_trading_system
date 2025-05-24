# Quant_trading_system

An end-to-end modular quantitative trading system powered by:
- 🧠 C++ core data processing with PyBind11
- 🤖 Reinforcement Learning with PPO / DQN (Stable-Baselines3)
- 📈 Custom trading environment based on gym.Env
- 🧩 Modular structure: alpha_engine / execution_engine / risk_control / env
- ✅ Integrated backtesting & real-time simulation pipeline

## Project Structure
- `cpp_core/` — C++ module with PyBind11 bindings
- `scripts/` — Training, evaluation, inference entry points
- `env/` — RL training environment (`env_cxx.py`, coming soon)
- `models/` — Trained agent files (DQN, PPO)
- `execution_engine/` — Simulated portfolio + order execution
- `risk_control/` — Placeholder for future stop-loss/vol control

## Quick Start

```bash
git clone https://github.com/kevinlmf/Quant_trading_system.git
cd Quant_trading_system

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd pybuild
python setup.py build_ext --inplace

python scripts/train_ppo.py

