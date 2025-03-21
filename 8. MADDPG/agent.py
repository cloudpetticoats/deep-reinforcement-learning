import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from collections import deque


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, o, a):
        x = torch.cat((o, a), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, obs, combi_obs, combi_actions, rewards, next_obs, combi_next_obs, dones):
        self.memory.append((obs, combi_obs, combi_actions, rewards, next_obs, combi_next_obs, dones))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MADDPG:
    def __init__(self, actor_input_dims, actor_output_dim, critic_input_dim, agent_num, lr_actor=1e-2, lr_critic=1e-3, gamma=0.99, tau=0.01, replay_buffer_capacity=8192, batch_size=64):
        self.agent_num = agent_num
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Initialize each agent's actor network and target actor network
        self.actors = [Actor(actor_input_dims[i], actor_output_dim[i], 64) for i in range(agent_num)]
        self.target_actors = [Actor(actor_input_dims[i], actor_output_dim[i], 64) for i in range(agent_num)]
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=lr_actor) for i in range(agent_num)]

        # Initialize each agent's critic network and target critic network
        self.critics = [Critic(critic_input_dim, 64) for _ in range(agent_num)]
        self.target_critics = [Critic(critic_input_dim, 64) for _ in range(agent_num)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=lr_critic) for i in range(agent_num)]

        # Initialize replay buffer for every agent
        self.replay_buffers = [ReplayBuffer(replay_buffer_capacity) for _ in range(agent_num)]

        self.update_target_networks(tau=1.0)

        self.loss_actors = [[] for _ in range(agent_num)]
        self.loss_critics = [[] for _ in range(agent_num)]

    def update_target_networks(self, tau=None):
        # soft update
        tau = self.tau if tau is None else tau
        for i in range(self.agent_num):
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def choose_action(self, observation, agent_idx):
        obs = torch.FloatTensor(observation).unsqueeze(0)
        return self.actors[agent_idx](obs).detach().numpy()[0]

    def update(self, target_train_flag):
        if len(self.replay_buffers[0]) < self.batch_size:
            return

        for idx, replay_buffer in enumerate(self.replay_buffers):
            obs, combi_obs, combi_actions, rewards, next_obs, combi_next_obs, terminateds = zip(*replay_buffer.sample(self.batch_size))
            combi_obs_1_dim = [[elem for sublist in item for elem in sublist] for item in combi_obs]
            combi_obs_1_dim = torch.FloatTensor(combi_obs_1_dim)
            combi_actions_1_dim = [[elem for sublist in item for elem in sublist] for item in combi_actions]
            combi_actions_1_dim = torch.FloatTensor(combi_actions_1_dim)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            combi_next_obs_1_dim = [[elem for sublist in item for elem in sublist] for item in combi_next_obs]
            combi_next_obs_1_dim = torch.FloatTensor(combi_next_obs_1_dim)
            terminateds = torch.FloatTensor(terminateds).unsqueeze(1)

            # update critic network
            combi_next_action = []
            for i in range(self.agent_num):
                temp_in = [combi_next_obs[j][i] for j in range(self.batch_size)]
                temp_in = torch.FloatTensor(np.array(temp_in))
                next_action_i = self.target_actors[i](temp_in).detach()
                combi_next_action.append(next_action_i)
            combi_next_action_1_dim = torch.hstack(combi_next_action)
            target_Q = self.target_critics[idx](combi_next_obs_1_dim, combi_next_action_1_dim).detach()
            target_Q = rewards + self.gamma * target_Q * (1 - terminateds)
            Q = self.critics[idx](combi_obs_1_dim, combi_actions_1_dim)
            critic_loss = nn.MSELoss()(Q, target_Q)
            self.loss_critics[idx].append(critic_loss.detach().numpy())
            self.critic_optimizers[idx].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[idx].step()

            # update actor network
            combi_action = []
            for i in range(self.agent_num):
                temp_in = [combi_obs[j][i] for j in range(self.batch_size)]
                temp_in = torch.FloatTensor(np.array(temp_in))
                action_i = self.actors[i](temp_in)
                combi_action.append(action_i)
            combi_action_1_dim = torch.hstack(combi_action)
            actor_loss = -self.critics[idx](combi_obs_1_dim, combi_action_1_dim).mean()
            self.loss_actors[idx].append(actor_loss.detach().numpy())
            self.actor_optimizers[idx].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[idx].step()

            if target_train_flag:
                self.update_target_networks()
