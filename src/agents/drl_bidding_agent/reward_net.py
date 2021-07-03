import torch
from src.agents.models.nn_model import MLPNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RewardNetAgent:
    def __init__(self, state_action_size, reward_size=1):
        self._reset_episode()
        self.reward_nn = MLPNetwork(state_size=state_action_size, action_size=reward_size, seed=0)

    def _reset_episode(self):
        self.V = 0.0
        self.S = []

    def perform_mini_batch(self):
        pass

    def _update_episode(self):
        pass

    def add_pair_t(self, state, action):
        self.S.append((state, action))

    def add_reward_t(self, reward_t):
        self.V += reward_t

    def get_reward_net_r_t(self, state_action_pair):
        """

        :param state_action_pair:
        :return:
        """
        sa = torch.from_numpy(state_action_pair).float().unsqueeze(0).to(device)
        return self.reward_nn(sa)
