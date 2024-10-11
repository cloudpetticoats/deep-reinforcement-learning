import torch
import numpy as np
import random
from pettingzoo.mpe import simple_adversary_v3

from agent import MADDPG
from utils import plot


# Hyperparameters
MAX_EPISODES = 1000  # 最大训练回合数
MAX_STEP = 25       # 每个回合中的最大步数
BATCH_SIZE = 128       # 训练批大小
MEMORY_CAPACITY = 1024  # 经验回放缓冲池容量
TARGET_INTERVAL = 20
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001
EPSILON_START = 1
EPSILON_END = 0.02
EPSILON_DECAY = MAX_EPISODES*MAX_STEP/2
GAMMA = 0.95


if __name__ == '__main__':
    env = simple_adversary_v3.parallel_env(continuous_actions=True)
    env.reset()
    agent_num = env.num_agents  # 智能体数量
    obs_dim = []  # 每个智能体的观测空间维度
    action_dim = []  # 每个智能体的动作空间维度
    reward_history = []
    agent_name_list = []

    for agent_idx, agent_name in enumerate(env.agents):
        obs_dim.append(env.observation_space(agent_name).shape[0])
        action_dim.append(env.action_space(agent_name).shape[0])
        agent_name_list.append(agent_name)

    maddpg = MADDPG(actor_input_dims=obs_dim, actor_output_dim=action_dim, critic_input_dim=sum(obs_dim)+sum(action_dim),
                    agent_num=agent_num, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, replay_buffer_capacity=MEMORY_CAPACITY,
                    batch_size=BATCH_SIZE, gamma=GAMMA)

    for episode_i in range(MAX_EPISODES):
        observations, infos = env.reset()
        episode_reward = 0  # 记录本回合奖励

        for step_i in range(MAX_STEP):
            actions = {}  # 初始化动作字典
            for agent_idx, agent_name in enumerate(env.agents):
                # choose action for every agent by epsilon-greedy
                epsilon = np.interp(x=episode_i * MAX_STEP + step_i, xp=[0, EPSILON_DECAY],
                                    fp=[EPSILON_START, EPSILON_END])
                random_sample = random.random()
                if random_sample <= epsilon:
                    actions = {agent_name: env.action_space(agent_name).sample() for agent_idx, agent_name in enumerate(env.agents)}
                else:
                    actions[agent_name] = maddpg.choose_action(observations[agent_name], agent_idx)

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            combine_observations = []
            combine_next_observations = []
            combine_actions = []
            for name_i in agent_name_list:
                combine_observations.append(observations[name_i])
                combine_next_observations.append(next_observations[name_i])
                combine_actions.append(actions[name_i])

            for agent_idx, agent_name in enumerate(env.agents):
                maddpg.replay_buffers[agent_idx].add(observations[agent_name], combine_observations, combine_actions,
                                                     rewards[agent_name], next_observations[agent_name],
                                                     combine_next_observations, terminations[agent_name])

            maddpg.update((episode_i*MAX_STEP+step_i)/TARGET_INTERVAL == 0)

            episode_reward += sum(rewards.values())
            observations = next_observations

            if all(terminations.values()):
                break

        reward_history.append(episode_reward)
        if episode_i%20 == 0:
            print(f"Episode {episode_i}/{MAX_EPISODES} | Episode total reward: {episode_reward}")

    # save model
    for agent_idx, agent_name in enumerate(env.agents):
        torch.save(maddpg.actors[agent_idx], f"actor_{agent_name}.pth")

    plot(maddpg.loss_actors, maddpg.loss_critics, reward_history)

    env.close()
