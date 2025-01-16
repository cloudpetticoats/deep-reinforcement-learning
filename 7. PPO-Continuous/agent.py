from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, max_action):
        super(PolicyNet, self).__init__()
        self.f1 = nn.Linear(input_dim, 64)
        self.f2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, output_dim)
        self.log_std = nn.Parameter(torch.zeros(1, output_dim))  # trick: Independent std
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        mean = torch.tanh(self.mean(x)) * self.max_action
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)
        return mean, std


class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.f1 = nn.Linear(input_dim, 64)
        self.f2 = nn.Linear(64, 64)
        self.f3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)


class TrajectoryMemory:
    def __init__(self, max_length):
        self.trajectory = deque(maxlen=max_length)

    def add(self, state, action, next_state, reward, done):
        self.trajectory.append((state, action, next_state, reward, done))

    def clear(self):
        self.trajectory.clear()

    def sample(self):
        return self.trajectory


class Agent:
    def __init__(self, input_dim, output_dim, max_trajectory_length, actor_lr, critic_lr, gamma, lamda, epoch, eps, max_action):
        self.actor = PolicyNet(input_dim, output_dim, max_action)
        self.critic = ValueNet(input_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.trajectory = TrajectoryMemory(max_trajectory_length)

        self.gamma = gamma
        self.lamda = lamda
        self.epoch = epoch
        self.eps = eps
        self.max_action = max_action

        self.actor_loss = []
        self.critic_loss = []

    def get_action(self, state):
        with torch.no_grad():
            mean, std = self.actor(torch.FloatTensor(state).unsqueeze(0))
            gaussian_distribution = torch.distributions.Normal(mean, std)
            a = gaussian_distribution.sample()
            a = torch.clamp(a, -self.max_action, self.max_action)
        return np.array([a.item()])

    def update(self):
        states, actions, next_states, rewards, dones = zip(*self.trajectory.sample())

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        with torch.no_grad():
            # computing TD-error
            target_val = self.critic(next_states)
            v_target_val = rewards + self.gamma * target_val * (1 - dones)
            v_val = self.critic(states)
            td_error = v_target_val - v_val

            # computing GAE
            advantages = torch.zeros_like(td_error)
            advantage = 0
            for i in reversed(range(len(td_error))):
                advantage = td_error[i] + self.gamma * self.lamda * advantage * (1 - dones[i])
                advantages[i] = advantage
            # trick: normalized GAE
            mean = advantages.mean()
            std = advantages.std()
            advantages = (advantages - mean) / std

        # computing old distribution
        mean, std = self.actor(states)
        gaussian_distribution = torch.distributions.Normal(mean.detach(), std.detach())
        old_log_distribution = gaussian_distribution.log_prob(actions)

        # train epoch times
        for epoch_i in range(self.epoch):
            # computing new distribution
            mean, std = self.actor(states)
            gaussian_distribution = torch.distributions.Normal(mean, std)
            log_distribution = gaussian_distribution.log_prob(actions)
            ratio_distribution = torch.exp(log_distribution - old_log_distribution)

            term1 = ratio_distribution * advantages.detach()
            term2 = torch.clamp(ratio_distribution, 1 - self.eps, 1 + self.eps) * advantages.detach()
            new_v_val = self.critic(states)

            actor_loss = torch.mean(-torch.min(term1, term2))
            critic_loss = torch.mean(F.mse_loss(new_v_val, v_target_val.detach()))

            self.actor_loss.append(actor_loss.item())
            self.critic_loss.append(critic_loss.item())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def plot(self, data, x_label):
        plt.plot(range(len(data)), data, color='r')
        plt.xlabel(x_label)
        plt.ylabel('loss')
        plt.show()
