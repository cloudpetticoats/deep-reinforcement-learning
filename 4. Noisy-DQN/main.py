import os.path
import random

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from agent import Agent

MAX_EPISODE = 300
MAX_STEP = 100
EPSILON_START = 1
EPSILON_END = 0.02

reward_list = []


if __name__ == '__main__':
    """
    ～～～ 注意：Noisy-DQN 不收敛，目前代码有问题，后续再完善，现在请不要使用～～～
    """

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = Agent(state_dim, action_dim)

    for episode_i in range(MAX_EPISODE):
        state, info = env.reset()
        episode_reward = 0

        agent.q_net.f1.reset_epsilon()
        agent.target_q_net.f1.reset_epsilon()
        agent.q_net.f2.reset_epsilon()
        agent.target_q_net.f2.reset_epsilon()
        agent.q_net.f3.reset_epsilon()
        agent.target_q_net.f3.reset_epsilon()
        agent.q_net.f4.reset_epsilon()
        agent.target_q_net.f4.reset_epsilon()

        for step_i in range(MAX_STEP):
            # epsilon greedy
            # epsilon = np.interp(x=episode_i*MAX_STEP+step_i, xp=[0, MAX_EPISODE*MAX_STEP/2],
            #                     fp=[EPSILON_START, EPSILON_END])
            # random_sample = random.random()
            # if random_sample <= epsilon:
            #     action = env.action_space.sample()
            # else:
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, info_ = env.step(action)

            agent.memory.push(state, action, next_state, reward, terminated or truncated)

            state = next_state
            episode_reward += reward

            agent.update()
            if terminated or truncated:
                break

        print(f"Episode: {episode_i+1}, Reward: {round(episode_reward, 3)}")
        reward_list.append(episode_reward)

    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = current_path + '/models/'
    timestamp = time.strftime("%Y%m%d%H%M%S")
    torch.save(agent.q_net.state_dict(), model_path + f"noisy_dqn_q_net_{timestamp}.pth")
    env.close()

    plt.plot(range(len(reward_list)), reward_list, color='b')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
