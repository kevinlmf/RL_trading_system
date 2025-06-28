import numpy as np

class BasicReward:
    def __init__(self):
        pass

    def __call__(self, state_vec, action, env):
        price_now = state_vec[0]
        price_prev = env.df.iloc[env.current_step - 1]["Close"]

        ret = (price_now - price_prev) / price_prev

        if env.position == 1:
            reward = ret
        elif env.position == -1:
            reward = -ret
        else:
            reward = 0.0

        return reward

