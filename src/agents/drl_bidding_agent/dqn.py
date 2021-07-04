import random
import numpy as np
from collections import deque, namedtuple
from src.agents.models.nn_model import MLPNetwork

import torch.optim as optim
import torch.nn.functional as F
import torch

LR = 1e-3
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 32
GAMMA = 1.0
TAU = 1e-3
UPDATE_FREQ = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self, state_size, action_size, seed=0):
        self.action_size = action_size
        random.seed(seed)

        # Q functions
        self.mlp_nn = MLPNetwork(state_size=state_size, action_size=action_size, seed=seed)
        self.mlp_nn_target = MLPNetwork(state_size=state_size, action_size=action_size, seed=seed)
        self.optimizer = optim.Adam(self.mlp_nn.parameters(), lr=LR)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # Internal step counter
        self._step_cnt = 0

    def act(self, state, eps=0.):
        """
        Choose action(scale factor beta_t) for given state
        :param state:
        :param eps:
        :return:
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.mlp_nn.eval()
        with torch.no_grad():
            action_values = self.mlp_nn(state)
        # Set neural net back to training mode
        self.mlp_nn.train()

        # return epsilon-greedy policy action
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update(self, s, a, r, next_s, terminal_state):
        """
        This method stores experience in the replay buffer,
        samples mini batch and applies gradient descent to update Q function,
        :param s:
        :param a:
        :param r:
        :param next_s:
        :param terminal_state:
        :return:
        """
        self.replay_buffer.add(s, a, r, next_s, terminal_state)

        # Update network every UPDATE_FREQ
        self._step_cnt = (self._step_cnt + 1) % UPDATE_FREQ

        if self._step_cnt == 0:
            experiences = self.replay_buffer.sample_mini_batch()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """

        :return:
        """
        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.mlp_nn_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + gamma*q_targets_next*(1-dones)

        q_expected = self.mlp_nn(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        print('DQN loss = ', loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.mlp_nn, self.mlp_nn_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        """
        Replay buffer for dqn agent
        :param buffer_size:
        :param batch_size:
        :param seed:
        """
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        e = self.experience(state, action, reward, next_state, done)
        self.replay_buffer.append(e)

    def sample_mini_batch(self):
        """

        :return:
        """
        k = min(self.batch_size, len(self.replay_buffer))
        experiences = random.sample(self.replay_buffer, k=k)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones
