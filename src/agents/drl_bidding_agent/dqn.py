from src.agents.models.nn_model import MLPNetwork


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.mlp_nn = MLPNetwork(state_size=state_size, action_size=action_size)

    # TODO: implement
    def act(self, state, eps=0.):
        """
        Choose action for given state
        :param state:
        :param eps:
        :return:
        """
        return 3

    def update(self, s, a, r, next_s, terminal_state):
        """
        This method
        :param s:
        :param a:
        :param r:
        :param next_s:
        :param terminal_state:
        :return:
        """
        pass




class ReplayBuffer:
    def __init__(self):
        pass