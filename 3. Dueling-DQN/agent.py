import random

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim

from collections import deque


class Q_net(nn.Module):
    """
    Dueling DQN only modifies the network architecture here compared to DQN
    """
    def __init__(self, input_dim, output_dim):
        super(Q_net, self).__init__()
        self.f1 = nn.Linear(input_dim, 128)
        self.f2 = nn.Linear(128, 128)

        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        a_val = self.A(x)
        v_val = self.V(x)
        return v_val + (a_val - a_val.mean(dim=1, keepdim=True))


class Memory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def store_transition(self, state, action, next_state, reward, end):
        self.memory.append((state, action, next_state, reward, end))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class Agent:
    def __init__(self, state_dim, action_dim, lr, buffer_size, batch_size, gamma, update_interval):
        self.q_net = Q_net(state_dim, action_dim)
        self.target_q_net = Q_net(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_net_optimizer = optim.AdamW(self.q_net.parameters(), lr=lr, amsgrad=True)
        self.memory = Memory(buffer_size)

        self.batch_size = batch_size
        self.gamma = gamma
        self.update_interval = update_interval
        self.update_count = 0

    def get_action(self, state):
        with torch.no_grad():
            return self.q_net(torch.tensor(state).unsqueeze(0))[0].max(0).indices.item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, next_states, rewards, ends = zip(*self.memory.sample(self.batch_size))
        states = torch.FloatTensor(np.array(states))
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        ends = torch.FloatTensor(ends).unsqueeze(1)

        # compute q value
        q = self.q_net(states).gather(1, actions)

        # compute target_q value
        with torch.no_grad():
            max_target_q_net = self.target_q_net(next_states).max(1).values.unsqueeze(1)
            target_q = rewards + (1 - ends) * self.gamma * max_target_q_net

        loss = nn.MSELoss()(q, target_q)
        self.q_net_optimizer.zero_grad()
        loss.backward()
        self.q_net_optimizer.step()

        self.update_count += 1
        # update target q network
        if self.update_count % self.update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
