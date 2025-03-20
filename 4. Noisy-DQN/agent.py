import random

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim

from collections import deque


class Q_net(nn.Module):
    def __init__(self, input_dim, output_dim, mode='Train'):
        super(Q_net, self).__init__()
        self.f1 = NoisyLinear(input_dim, 128, mode)
        self.f2 = NoisyLinear(128, 128, mode)
        self.f3 = NoisyLinear(128, 128, mode)
        self.f4 = NoisyLinear(128, output_dim, mode)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        x = torch.relu(self.f3(x))
        return self.f4(x)


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, mode):
        super(NoisyLinear, self).__init__()
        self.weight_mu = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.bias_mu = nn.Parameter(torch.FloatTensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_dim))

        self.register_buffer('weight_epsilon', torch.FloatTensor(output_dim, input_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_dim))

        self.mode = mode
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.reset_epsilon()

    def forward(self, x):
        if self.mode == 'Train':
            return nn.functional.linear(x, self.weight_mu + self.weight_sigma.mul(self.weight_epsilon),
                                        self.bias_mu + self.bias_sigma.mul(self.bias_epsilon))
        else:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)

    def reset_epsilon(self):
        self.weight_epsilon = torch.rand(self.output_dim, self.input_dim)
        self.bias_epsilon = torch.rand(self.output_dim)


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
            return self.q_net(torch.tensor(state)).max(0).indices.item()

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
