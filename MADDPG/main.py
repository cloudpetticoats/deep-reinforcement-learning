import torch
import numpy as np
import random
from pettingzoo.mpe import simple_adversary_v3
import matplotlib.pyplot as plt

from DDPG.agent import LR_ACTOR, LR_CRITIC
from agent import MADDPG


def plot(actor_loss, critic_loss, reward):
    # 创建子图
    fig, axs = plt.subplots(3, 3, figsize=(25, 20))

    axs[0, 0].plot(range(len(actor_loss[0])), actor_loss[0], color='purple')
    axs[0, 0].set_xlabel('step')
    axs[0, 0].set_ylabel('adversary_actor loss')

    axs[1, 0].plot(range(len(actor_loss[1])), actor_loss[1], color='purple')
    axs[1, 0].set_xlabel('step')
    axs[1, 0].set_ylabel('agent0_actor loss')

    axs[2, 0].plot(range(len(actor_loss[2])), actor_loss[2], color='purple')
    axs[2, 0].set_xlabel('step')
    axs[2, 0].set_ylabel('agent1_actor loss')

    axs[0, 1].plot(range(len(critic_loss[0])), critic_loss[0], color='purple')
    axs[0, 1].set_xlabel('step')
    axs[0, 1].set_ylabel('adversary_critic loss')

    axs[1, 1].plot(range(len(critic_loss[1])), critic_loss[1], color='purple')
    axs[1, 1].set_xlabel('step')
    axs[1, 1].set_ylabel('agent0_critic loss')

    axs[2, 1].plot(range(len(critic_loss[2])), critic_loss[2], color='purple')
    axs[2, 1].set_xlabel('step')
    axs[2, 1].set_ylabel('agent_1_critic loss')

    axs[0, 2].plot(range(len(reward)), reward, color='purple')
    axs[0, 2].set_xlabel('episode')
    axs[0, 2].set_ylabel('episode reward')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Hyperparameters
MAX_EPISODES = 300  # 最大训练回合数
MAX_STEP = 25       # 每个回合中的最大步数
BATCH_SIZE = 256       # 训练批大小
MEMORY_CAPACITY = 4096  # 经验回放缓冲池容量
TARGET_INTERVAL = 60
LR_ACTOR = 0.01
LR_CRITIC = 0.01
EPSILON_START = 1
EPSILON_END = 0.02
EPSILON_DECAY = MAX_EPISODES*MAX_STEP/2


if __name__ == '__main__':
    env = simple_adversary_v3.parallel_env(continuous_actions=True)
    env.reset()
    agent_num = env.num_agents  # 智能体数量
    obs_dim = []  # 每个智能体的观测空间维度
    action_dim = []  # 每个智能体的动作空间维度
    reward_history = []

    for agent_idx, agent_name in enumerate(env.agents):
        obs_dim.append(env.observation_space(agent_name).shape[0])
        action_dim.append(env.action_space(agent_name).shape[0])

    maddpg = MADDPG(actor_input_dims=obs_dim, actor_output_dim=action_dim, critic_input_dim=[a + b for a, b in zip(obs_dim, action_dim)],
                    agent_num=agent_num, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, replay_buffer_capacity=MEMORY_CAPACITY, batch_size=BATCH_SIZE)

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

            for agent_idx, agent_name in enumerate(env.agents):
                maddpg.replay_buffers[agent_idx].add(observations[agent_name], actions[agent_name], rewards[agent_name], next_observations[agent_name], terminations[agent_name])

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
