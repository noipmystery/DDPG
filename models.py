from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


def fan_in(size):
    siz = size[0]
    v = 1. / np.sqrt(siz)
    return torch.Tensor(size).uniform_(-v, v).float()


class Actor(nn.Module):
    def init_weights(self, init_w):
        self.fc1.weight.data = fan_in(self.fc1.weight.data.size())
        self.fc2.weight.data = fan_in(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w).float()

    def __init__(self, state_dim, action_dim, action_bound, hidden_1=400, hidden_2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, action_dim)
        self.action_bound = action_bound
        self.init_weights(init_w)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tanh(x) * self.action_bound


class Critic(nn.Module):
    def init_weights(self, init_w):
        self.fc1.weight.data = fan_in(self.fc1.weight.data.size())
        self.fc2.weight.data = fan_in(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w).float()

    def __init__(self, state_dim, action_dim, hidden_1=400, hidden_2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1 + action_dim, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 1)
        self.init_weights(init_w)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], 1)
        print(type(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
