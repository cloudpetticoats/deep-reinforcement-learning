import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim


# Hyperparameters
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
MEMORY_SIZE = 100000
BATCH_SIZE = 64
TAU = 5e-3


class Actor(nn.Module):
    def __init__(self, state_num, action_num, hidden_num=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_num, hidden_num)
        self.fc2 = nn.Linear(hidden_num, hidden_num)
        self.fc3 = nn.Linear(hidden_num, action_num)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * 2
        return x


class Critic(nn.Module):
    def __init__(self, state_num, action_num, hidden_num=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_num+action_num, hidden_num)
        self.fc2 = nn.Linear(hidden_num, hidden_num)
        self.fc3 = nn.Linear(hidden_num, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_experience(self, state, action, reward, next_state, terminated):
        # expand become [1, 3]
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        state, action, reward, next_state, terminated = zip(*random.sample(self.buffer, batch_size))
        # TODO ? state到底是什么样子的需要np.concatenate
        return np.concatenate(state), action, reward, np.concatenate(next_state), terminated

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, state_num, action_num):
        self.actor = Actor(state_num, action_num)
        self.target_actor = Actor(state_num, action_num)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_num, action_num)
        self.target_critic = Critic(state_num, action_num)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayBuffer(BATCH_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        states, actions, rewards, next_states, terminateds = self.replay_buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        terminateds = torch.FloatTensor(terminateds)

        # update critic network
        next_action = self.target_actor(next_states)
        target_Q = self.target_critic(next_states, next_action.detach())
        target_Q = rewards + GAMMA * target_Q * (1 - terminateds)
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
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        # update target actor network
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
