from collections import deque, namedtuple
import random
import torch.nn as nn
import torch
import torch.optim as optim

# Hyperparameters
LR = 1e-4
MEMORY_SIZE = 10000

# 定义经验的元组格式
experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


class Q_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Q_net, self).__init__()
        self.f1 = nn.Linear(input_dim, 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        return self.f3(x)


class Memory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        self.memory.append(experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class Agent:
    def __init__(self, state_dim, action_dim):

        self.q_net = Q_net(state_dim, action_dim)
        self.target_q_net = Q_net(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_net_optimizer = optim.AdamW(self.q_net.parameters(), lr=LR, amsgrad=True)

        self.memory = Memory(MEMORY_SIZE)

    def get_action(self, state):
        # 禁用自动梯度计算
        with torch.no_grad():
            return self.q_net(state).max(1).indices.view(1, 1)
