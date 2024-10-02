import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque


class Actor(nn.Module):
    # Actor网络，用于生成确定性动作
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))  # 输出动作，通过softmax limited
        return x


class Critic(nn.Module):
    # Critic网络，用于估计Q值
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 最后一层输出Q值
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, obs, actions, rewards, next_obs, dones):
        self.memory.append((obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MADDPG:
    def __init__(self, actor_input_dims, actor_output_dim, critic_input_dim, agent_num, lr_actor=1e-2, lr_critic=1e-3, gamma=0.99, tau=0.01, replay_buffer_capacity=8192, batch_size=64):
        self.agent_num = agent_num  # 智能体数量
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # 初始化每个智能体的actor网络和target_actor网络
        self.actors = [Actor(actor_input_dims[i], actor_output_dim[i], 64) for i in range(agent_num)]
        self.target_actors = [Actor(actor_input_dims[i], actor_output_dim[i], 64) for i in range(agent_num)]
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=lr_actor) for i in range(agent_num)]

        # 初始化每个智能体的critic网络和target_critic网络
        self.critics = [Critic(critic_input_dim[i], 64) for i in range(agent_num)]
        self.target_critics = [Critic(critic_input_dim[i], 64) for i in range(agent_num)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=lr_critic) for i in range(agent_num)]

        # initialize replay buffer for every agent
        self.replay_buffers = [ReplayBuffer(replay_buffer_capacity)] * agent_num

        self.update_target_networks(tau=1.0)

        self.loss_actors = []
        self.loss_critics = []

    def update_target_networks(self, tau=None):
        # 软更新目标网络
        tau = self.tau if tau is None else tau
        for i in range(self.agent_num):
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def choose_action(self, observation, agent_idx):
        obs = torch.FloatTensor(observation).unsqueeze(0)
        return self.actors[agent_idx](obs).detach().numpy()[0]  # 获取动作并转为numpy

    def update(self):
        if len(self.replay_buffers[0]) < self.batch_size:
            return

        for idx, replay_buffer in enumerate(self.replay_buffers):
            obs, actions, rewards, next_obs, terminateds = zip(*replay_buffer.sample(self.batch_size))
            obs = torch.FloatTensor(obs)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_obs = torch.FloatTensor(next_obs)
            terminateds = torch.FloatTensor(terminateds).unsqueeze(1)

            # update critic network
            next_action = self.target_actors[idx](next_obs)
            target_Q = self.target_critics[idx](next_obs, next_action.detach())
            target_Q = rewards + self.gamma * target_Q * (1 - terminateds)
            Q = self.critics[idx](obs, actions)
            critic_loss = nn.MSELoss()(Q, target_Q)
            self.loss_critics[idx].append(critic_loss.detach().numpy())
            self.critic_optimizers[idx].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[idx].step()

            # update actor network
            actor_loss = -self.critics[idx](obs, self.actors[idx](obs)).mean()
            self.loss_actors[idx].append(actor_loss.detach().numpy())
            self.actor_optimizers[idx].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[idx].step()

            self.update_target_networks()
