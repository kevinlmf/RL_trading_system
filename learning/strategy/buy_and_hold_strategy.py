class BuyAndHoldStrategy:
    """
    A simple buy-and-hold strategy:
    - Buys once at the beginning of the episode.
    - Holds the position for the rest of the episode.
    """

    def __init__(self, env):
        self.env = env
        self.has_bought = False

    def select_action(self, state):
        """
        Selects the action for the current timestep.
        Returns:
            int: 1 for 'buy/hold', 0 for 'do nothing'
        """
        if not self.has_bought:
            self.has_bought = True
            return 1  # Buy at the first opportunity
        return 1      # Continue holding the position
