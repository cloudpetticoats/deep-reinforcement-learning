import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from collections import deque


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(state_dim, 64)
        self.f2 = nn.Linear(64, 64)
        self.f3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        x = torch.tanh(self.f3(x)) * 2
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.f1 = nn.Linear(state_dim + action_dim, 64)
        self.f2 = nn.Linear(64, 64)
        self.f3 = nn.Linear(64, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        x = self.f3(x)
        return x


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
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, buffer_size, batch_size, gamma, update_interval, tau):
        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = Memory(buffer_size)

        self.batch_size = batch_size
        self.gamma = gamma
        self.update_interval = update_interval
        self.tau = tau

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, next_states, rewards, ends = zip(*self.memory.sample(self.batch_size))
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.vstack(actions))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        ends = torch.FloatTensor(ends).unsqueeze(1)

        # update critic network
        next_action = self.target_actor(next_states)
        target_Q = self.target_critic(next_states, next_action.detach())
        target_Q = rewards + self.gamma * target_Q * (1 - ends)
        Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor network
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target critic network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # update target actor network
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
