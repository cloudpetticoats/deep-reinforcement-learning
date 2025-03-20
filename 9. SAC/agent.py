import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from collections import deque
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(state_dim, 64)
        self.f2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.f1(state))
        x = F.relu(self.f2(x))
        mu = self.max_action * torch.tanh(self.mu(x))
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # q1
        self.q1_f1 = nn.Linear(state_dim + action_dim, 64)
        self.q1_f2 = nn.Linear(64, 64)
        self.q1_f3 = nn.Linear(64, 1)
        # q2
        self.q2_f1 = nn.Linear(state_dim + action_dim, 64)
        self.q2_f2 = nn.Linear(64, 64)
        self.q2_f3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        # q1
        q1 = F.relu(self.q1_f1(x))
        q1 = F.relu(self.q1_f2(q1))
        q1 = self.q1_f3(q1)
        # q2
        q2 = F.relu(self.q2_f1(x))
        q2 = F.relu(self.q2_f2(q2))
        q2 = self.q2_f3(q2)
        return q1, q2


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def store_transition(self, state, action, next_state, reward, end):
        self.memory.append((state, action, next_state, reward, end))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class SAC_Agent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr, entropy_lr, buffer_size, batch_size, gamma, tau):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.entropy_optimizer = torch.optim.Adam([self.log_alpha], lr=entropy_lr)

        self.max_action = max_action
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def get_action(self, state):
        with torch.no_grad():
            mu, std = self.actor(torch.FloatTensor(state))
            dist = Normal(mu, std)
            action = dist.sample()
            action = torch.clamp(action, -self.max_action, self.max_action)
        return action.detach().numpy()

    def get_action_re_parameter(self, state):
        mu, std = self.actor(torch.FloatTensor(state))
        dist = Normal(mu, std)
        noise = Normal(0, 1)
        n = noise.sample()
        action = torch.tanh(mu + std * n)
        action = torch.clamp(action, -self.max_action, self.max_action)
        action_log_prob = dist.log_prob(mu + std * n) - torch.log(1 - action.pow(2) + 1e-6)
        return action, action_log_prob

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, next_states, rewards, ends = zip(*self.replay_buffer.sample(self.batch_size))
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        ends = torch.FloatTensor(ends).unsqueeze(1)

        # update critic
        new_actions, new_action_log_probs = self.get_action_re_parameter(next_states)
        target_q1, target_q2 = self.target_critic(next_states, new_actions)
        target_q = rewards + self.gamma * (torch.min(target_q1, target_q2) - self.alpha * new_action_log_probs) * (1 - ends)
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q.detach()) + F.mse_loss(q2, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        n_actions, n_action_log_probs = self.get_action_re_parameter(states)
        q1, q2 = self.critic(states, n_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * n_action_log_probs - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update entropy
        alpha_loss = -(self.log_alpha.exp() * (n_action_log_probs + self.target_entropy).detach()).mean()
        self.entropy_optimizer.zero_grad()
        alpha_loss.backward()
        self.entropy_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # update target critic
        for target_params, params in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_params.data.copy_((1.0 - self.tau) * target_params.data + self.tau * params.data)
