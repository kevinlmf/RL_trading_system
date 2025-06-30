# 🔧 Formal Objective of Reinforcement Learning in Trading

In a reinforcement learning-based trading system, the agent interacts with a simulated market environment. At each time step, it observes a state, takes an action, and receives a reward. The goal is to learn a policy that maximizes the expected cumulative reward over time.

## 🔢 Mathematical Definition

The optimal policy \( \pi^* \) is defined as:

\[
\pi^*(s) = \arg\max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{T} \gamma^t r_t \mid \pi \right]
\]

### Where:

- \( s \): the current **market state**, e.g., past window of prices, indicators, or portfolio state
- \( \pi \): the **policy function**, mapping states to actions
- \( a = \pi(s) \): the **action** taken at state \( s \) (Buy, Sell, Hold)
- \( r_t \): the **immediate reward**, e.g., return or Sharpe-adjusted profit at time \( t \)
- \( \gamma \in [0, 1] \): the **discount factor**, quantifying the agent's preference for immediate vs. future rewards
- \( T \): the **episode horizon**, i.e., number of trading steps

## 🧠 Intuition

> The agent aims to make sequential trading decisions that lead to the highest long-term profitability, rather than short-term gains.

By maximizing the expected discounted return, the policy encourages sustainable and risk-aware trading strategies.


