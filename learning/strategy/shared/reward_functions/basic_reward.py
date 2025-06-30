def BasicReward(obs, action, env):
    price = env.df.iloc[env.current_step][env.price_column]
    prev_value = env.balance + env.shares_held * price
    new_value = env.balance + env.shares_held * price
    return (new_value - prev_value) / (prev_value + 1e-8)
