from agent import Agent
from normalization import Normalization, RewardScaling

import gym
import os.path
import torch
import matplotlib.pyplot as plt


# Hyperparameters
MAX_EPISODE = 800
MAX_TRAJECTORY_LENGTH = 200
ACTOR_LR = 3e-4
CRITIC_LR = 2e-3
GAMMA = 0.95
LAMBDA = 0.9
EPOCH = 15
EPS = 0.15
ENTROPY_COEF = 0.01


reward_list = []


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')  # render_mode="human"
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    # state_norm = Normalization(shape=n_states)
    scaling_reward = RewardScaling(shape=1, gamma=GAMMA)

    agent = Agent(n_states, n_actions, MAX_TRAJECTORY_LENGTH, ACTOR_LR,
                  CRITIC_LR, GAMMA, LAMBDA, EPOCH, EPS, max_action, ENTROPY_COEF)
    for i in range(MAX_EPISODE):
        state, info = env.reset()
        scaling_reward.reset()
        done = False
        reward_total = 0
        while not done:
            # state = state_norm(state)  # trick: state normalization
            action, a_log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward_total += reward
            # next_state = state_norm(next_state)  # trick: state normalization
            reward = scaling_reward(reward)  # trick: scaling reward
            done = terminated | truncated
            agent.trajectory.add(state, action, a_log_prob, next_state, reward, done)
            state = next_state

        reward_list.append(reward_total)

        agent.update()
        agent.trajectory.clear()
        print(f"Episode：{i + 1}, Reward：{reward_total}")

    agent.plot(agent.actor_loss, 'actor_epoch')
    agent.plot(agent.critic_loss, 'critic_epoch')

    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = current_path + '/models/'
    torch.save(agent.actor.state_dict(), model_path + "ppo_actor_net.pth")
    env.close()

    plt.plot(range(len(reward_list)), reward_list, color='b')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
