import os.path
import gym
import numpy as np
import random
import argparse

from agent import Agent
from common.tools import plot_reward, save_model


def init_parameters():
    """
    Initialize the parameters required for the algorithm.
    """
    parser = argparse.ArgumentParser(description="DDPG Hyperparameters")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--episode_length", type=int, default=200, help="Maximum episodes")
    parser.add_argument("--step_length", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--epsilon_start", type=float, default=1, help="Initial value of epsilon in epsilon-greedy exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.02, help="Final value of epsilon in epsilon-greedy exploration")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Learning rate for actor network")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="Learning rate for critic network")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
    parser.add_argument("--interval", type=int, default=100, help="Training interval of the target network")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter for target network")
    arg = parser.parse_args()
    print(arg)
    return arg


if __name__ == '__main__':
    args = init_parameters()
    environment = gym.make(args.env_name)
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.shape[0]
    agent = Agent(state_dim, action_dim, args.actor_lr, args.critic_lr, args.buffer_size, args.batch_size, args.gamma, args.interval, args.tau)

    reward_list = []
    for episode_i in range(args.episode_length):
        state, info = environment.reset()
        episode_reward = 0

        for step_i in range(args.step_length):
            # epsilon greedy
            epsilon = np.interp(x=episode_i * args.step_length + step_i, xp=[0, args.episode_length * args.step_length / 2],
                                fp=[args.epsilon_start, args.epsilon_end])
            random_sample = random.random()
            if random_sample <= epsilon:
                action = np.random.uniform(low=-2, high=2, size=action_dim)
            else:
                action = agent.get_action(state)

            next_state, reward, terminated, truncated, info = environment.step(action)
            agent.memory.store_transition(state, action, next_state, reward, terminated or truncated)
            agent.update()

            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break

        reward_list.append(episode_reward)
        print(f"Episode: {episode_i+1}, Reward: {round(episode_reward, 3)}")

    # save model
    save_model(os.path.dirname(os.path.realpath(__file__)), agent.actor.state_dict(), 'ddpg_actor')
    # plot reward curve
    plot_reward(reward_list, 30)
