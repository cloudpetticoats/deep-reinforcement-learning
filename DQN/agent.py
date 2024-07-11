from collections import deque
import random
import torch.nn as nn
import torch
import torch.optim as optim

# Hyperparameters
LR = 1e-4
MEMORY_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99
UPDATE_TARGET = 100


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

    def push(self, state, action, next_state, reward, end):
        self.memory.append((torch.tensor(state), action, torch.tensor(next_state), reward, end))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class Agent:
    def __init__(self, state_dim, action_dim):

        self.q_net = Q_net(state_dim, action_dim)
        self.target_q_net = Q_net(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_net_optimizer = optim.AdamW(self.q_net.parameters(), lr=LR, amsgrad=True)

        self.memory = Memory(MEMORY_SIZE)

        self.update_count = 0

    def get_action(self, state):
        # 禁用自动梯度计算
        with torch.no_grad():
            return self.q_net(torch.tensor(state)).max(0).indices.item()

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # state :[ 0.00953062  0.04529195 -0.00640777 -0.035508  ]  action : 0
        states, actions, next_states, rewards, ends = zip(*self.memory.sample(BATCH_SIZE))
        states = torch.stack(states)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)
        next_states = torch.stack(next_states)
        ends = torch.FloatTensor(ends).unsqueeze(1)

        # q_net的Q值（动作状态函数）
        q = self.q_net(states).gather(1, actions)

        # target_q of target_q_net
        with torch.no_grad():
            max_target_q_net = self.target_q_net(next_states).max(1).values.unsqueeze(1)
            target_q = rewards + (1 - ends) * GAMMA * max_target_q_net

        loss = nn.MSELoss()(q, target_q)
        self.q_net_optimizer.zero_grad()
        loss.backward()
        self.q_net_optimizer.step()

        if self.update_count % UPDATE_TARGET == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
