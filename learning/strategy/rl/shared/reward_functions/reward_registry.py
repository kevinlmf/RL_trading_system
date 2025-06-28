from .basic_reward import BasicReward
# from .copula_risk_reward import CopulaRiskReward  # 如果你之后实现了

def get_reward_fn(reward_type):
    if reward_type == "basic":
        return BasicReward()
    # elif reward_type == "copula":
    #     return CopulaRiskReward()
    else:
        raise ValueError(f"❌ Unknown reward type: {reward_type}")

