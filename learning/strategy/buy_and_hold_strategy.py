class BuyAndHoldStrategy:
    def __init__(self, env):
        self.env = env
        self.has_bought = False

    def select_action(self, state):
        # 假设动作 1 是买入，0 是观望
        if not self.has_bought:
            self.has_bought = True
            return 1  # 第一次执行买入
        return 1  # 之后持续持仓
