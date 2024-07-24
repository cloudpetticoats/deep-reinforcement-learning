import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.f1 = nn.Linear(input_dim, 64)
        self.f2 = nn.Linear(64, 64)
        self.f3 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, output_dim)
        self.std = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        return F.tanh(self.mu(x)) * 2, F.softplus(self.std(x))


class ValueNet(nn.Module):
    """
    V(s) 状态价值函数
    """
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.f1 = nn.Linear(input_dim, 64)
        self.f2 = nn.Linear(64, 64)
        self.f3 = nn.Linear(64, 64)
        self.f4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        return self.f4(x)


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
    def __init__(self, input_dim, output_dim, max_trajectory_length, actor_lr, critic_lr, gamma, lamda, epoch, eps):
        self.actor = PolicyNet(input_dim, output_dim)
        self.critic = ValueNet(input_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.trajectory = TrajectoryMemory(max_trajectory_length)

        self.gamma = gamma
        self.lamda = lamda
        self.epoch = epoch
        self.eps = eps

        self.actor_loss = []
        self.critic_loss = []

    def get_action(self, state):
        with torch.no_grad():
            mu, std = self.actor(torch.tensor(state).unsqueeze(0))
            gaussian_distribution = torch.distributions.Normal(mu, std)
        return [gaussian_distribution.sample().item()]

    def update(self):
        states, actions, next_states, rewards, dones = zip(*self.trajectory.sample())

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # computing TD-error
        target_val = self.critic(next_states)
        v_target_val = rewards + self.gamma * target_val * (1 - dones)
        v_val = self.critic(states)
        td_error = v_val - v_target_val

        # computing GAE
        td_error = td_error.detach().numpy()
        advantages = []
        advantage = 0
        for td in td_error[::-1]:
            advantage = self.gamma * self.lamda * advantage + td
            advantages.append(advantage)
        advantages.reverse()
        advantages = torch.FloatTensor(np.array(advantages))

        # computing old distribution
        mu, std = self.actor(states)
        gaussian_distribution = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_distribution = gaussian_distribution.log_prob(actions)

        # train epoch times
        for epoch_i in range(self.epoch):
            # computing new distribution
            mu, std = self.actor(states)
            gaussian_distribution = torch.distributions.Normal(mu, std)
            log_distribution = gaussian_distribution.log_prob(actions)
            ratio_distribution = torch.exp(log_distribution - old_log_distribution)

            term1 = ratio_distribution * advantages
            term2 = torch.clamp(ratio_distribution, 1 - self.eps, 1 + self.eps) * advantages
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
