# 🧠 Quant_trading_system Setup Guide

This repo contains a modular RL trading system using Python + C++.

## ✅ 1. Setup Virtual Environment

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
\`\`\`

## ✅ 2. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## ✅ 3. Build PyBind11 C++ Module

\`\`\`bash
./setup.sh
\`\`\`

## ✅ 4. Run Tests

\`\`\`bash
python scripts/test_cpp_module.py
python scripts/test_random.py
\`\`\`

---

## 🧪 Optional: Train + Compare Strategies

\`\`\`bash
./scripts/run_training.sh ppo
python scripts/compare_strategies.py
\`\`\`

## ✅ Output

- Models saved to: \`models/\`
- Logs saved to: \`tensorboard/\`
