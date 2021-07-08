import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from src.agents.models.nn_model import MLPNetwork

from collections import defaultdict, deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # mini-batch size
LR = 1e-3


class RewardNetAgent:
    def __init__(self, state_action_size, reward_size=1, seed=0):
        random.seed(seed)
        self.V = 0.0
        self.S = []  # use array-like data structure since we are sure that each state in the sequence is unique

        # Init reward net
        self.reward_nn = MLPNetwork(state_size=state_action_size, action_size=reward_size, seed=0)
        self.optimizer = optim.Adam(self.reward_nn.parameters(), lr=LR)

        # Init reward dict
        self.M = defaultdict()

        # Init reply buffer
        self.replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0)

    def reset_episode(self):
        """
        Reset V and S at the end of each episode
        :return:
        """
        self.V = 0.0
        self.S = []

    def perform_mini_batch(self):
        """

        :return:
        """
        if len(self.replay_buffer) > 1:
            state_actions, rewards = self.replay_buffer.sample_mini_batct()

            # Get expected rewards from reward net
            expected_rewards = self.reward_nn(state_actions)

            # Compute loss
            loss = F.mse_loss(expected_rewards, rewards)
            print('Reward net loss: ', loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_episode(self):
        """

        :return:
        """
        for (s, a) in self.S:
            sa = tuple(np.append(s, a))
            # update reward dictionary
            max_r = max(self.V, self.M.get(sa, 0))
            # store state_action, reward pair to replay_buffer
            self.replay_buffer.add(sa, max_r)

    def add_pair_t(self, state, action):
        """

        :param state:
        :param action:
        :return:
        """
        self.S.append((state, action))

    def add_reward_t(self, reward_t):
        """

        :param reward_t:
        :return:
        """
        self.V += reward_t

    def get_reward_net_r_t(self, state_action_pair):
        """

        :param state_action_pair:
        :return:
        """
        sa = torch.from_numpy(state_action_pair).float().unsqueeze(0).to(device)
        return self.reward_nn(sa)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        """

        :param buffer_size:
        :param batch_size:
        :param seed:
        """
        random.seed(seed)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('experience', field_names=['state_action', 'reward'])

    def add(self, state_action, reward):
        """

        :param state_action:
        :param reward:
        :return:
        """
        e = self.experience(state_action, reward)
        self.replay_buffer.append(e)

    def sample_mini_batct(self):
        """

        :return:
        """
        k = min(self.batch_size, len(self.replay_buffer))
        experiences = random.sample(self.replay_buffer, k)

        state_actions = torch.from_numpy(np.vstack([e.state_action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)

        return state_actions, rewards

    def __len__(self):
        """

        :return:
        """
        return len(self.replay_buffer)
