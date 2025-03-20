import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import deque


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.f1 = nn.Linear(input_dim, 64)
        self.f2 = nn.Linear(64, 64)
        self.action_probs = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return F.softmax(self.action_probs(x), dim=-1)  # Output action probability


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

    def get_action(self, state):
        with torch.no_grad():
            action_probs = self.actor(torch.tensor(state).unsqueeze(0))
            distribution = torch.distributions.Categorical(probs=action_probs)
            action = distribution.sample()
        return action.item()  # return action index

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
        td_error = v_target_val - v_val

        # computing GAE
        advantages = torch.zeros_like(td_error)
        advantage = 0
        for i in reversed(range(len(td_error))):
            advantage = td_error[i] + self.gamma * self.lamda * advantage * (1 - dones[i])
            advantages[i] = advantage

        # computing old distribution
        action_probs = self.actor(states).detach()
        old_distribution = torch.distributions.Categorical(probs=action_probs)
        old_log_distribution = old_distribution.log_prob(actions)

        # train epoch times
        for epoch_i in range(self.epoch):
            # computing new distribution
            action_probs = self.actor(states)
            new_distribution = torch.distributions.Categorical(probs=action_probs)
            log_distribution = new_distribution.log_prob(actions)
            ratio_distribution = torch.exp(log_distribution - old_log_distribution)

            # computing loss
            term1 = ratio_distribution * advantages.detach()
            term2 = torch.clamp(ratio_distribution, 1 - self.eps, 1 + self.eps) * advantages.detach()
            new_v_val = self.critic(states)
            actor_loss = torch.mean(-torch.min(term1, term2))
            critic_loss = torch.mean(F.mse_loss(new_v_val, v_target_val.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
