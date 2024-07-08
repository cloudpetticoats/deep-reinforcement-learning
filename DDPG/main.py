import gym
import numpy as np
import random

from agent import Agent

environment = gym.make(id='Pendulum-v1')
STATE_NUM = environment.observation_space.shape[0]
ACTION_NUM = environment.action_space.shape[0]

agent = Agent(STATE_NUM, ACTION_NUM)

# Hyperparameters
NUM_EPISODE = 100
NUM_STEP = 200
EPSILON_START = 1
EPSILON_END = 0.02
EPSILON_DECAY = NUM_EPISODE*NUM_STEP/2

# process data
reward_buffer = np.empty(shape=NUM_EPISODE)

for episode_i in range(NUM_EPISODE):
    state, info = environment.reset()
    episode_reward = 0

    for step_i in range(NUM_STEP):
        # get action by epsilon_greedy method
        epsilon = np.interp(x=episode_i*NUM_STEP+step_i, xp=[0, EPSILON_DECAY], fp=[EPSILON_START, EPSILON_END])
        random_sample = random.random()
        if random_sample <= epsilon:
            action = np.random.uniform(low=-2, high=2, size=ACTION_NUM)
        else:
            action = agent.get_action(state)

        # perform the action
        next_state, reward, terminated, truncated, info = environment.step(action)

        # store experience into replayBuffer
        agent.ReplayBuffer.add_experience(state, action, reward, next_state, terminated)

        state = next_state
        episode_reward += reward

        agent.update()
        if terminated:
            break

    reward_buffer[episode_i] = episode_reward
    print(f"Episode: {episode_i+1}, Reward: {round(episode_reward, 3)}")

environment.close()
