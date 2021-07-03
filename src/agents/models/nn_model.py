import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, layer1_units=100, layer2_units=100, layer3_units=100):
        """
        MLP model for agent
        :param state_size:
        :param action_size:
        :param seed:
        :param layer1_units:
        :param layer2_units:
        :param layer3_units:
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_size, layer1_units)
        self.layer2 = nn.Linear(layer1_units, layer2_units)
        self.layer3 = nn.Linear(layer2_units, layer3_units)
        self.layer4 = nn.Linear(layer3_units, action_size)

    def forward(self, x):
        """
        Build MLP model
        :param x:
        :return:
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
