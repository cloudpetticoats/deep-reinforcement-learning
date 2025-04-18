import os.path
import random
import gym
import numpy as np
import argparse

from agent import Agent
from common.tools import plot_reward, save_model


def init_parameters():
    """
    Initialize the parameters required for the algorithm.
    """
    parser = argparse.ArgumentParser(description="Dueling-DQN Hyperparameters")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--episode_length", type=int, default=200, help="Maximum episodes")
    parser.add_argument("--step_length", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--epsilon_start", type=float, default=1, help="Initial value of epsilon in epsilon-greedy exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.02, help="Final value of epsilon in epsilon-greedy exploration")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training the Q-network")
    parser.add_argument("--buffer_size", type=int, default=10000, help="The size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size used for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
    parser.add_argument("--interval", type=int, default=100, help="Training interval of the target Q-network")

    parser.add_argument("--episode_length", type=int, default=200, help="Maximum episodes")
    parser.add_argument("--step_length", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--epsilon_start", type=float, default=1,
                        help="Initial value of epsilon in epsilon-greedy exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.02,
                        help="Final value of epsilon in epsilon-greedy exploration")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training the Q-network")
    parser.add_argument("--buffer_size", type=int, default=10000, help="The size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
    parser.add_argument("--interval", type=int, default=100, help="Training interval of the Q-network")

    arg = parser.parse_args()
    print(arg)
    return arg


if __name__ == '__main__':
    args = init_parameters()
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = Agent(state_dim, action_dim, args.lr, args.buffer_size, args.batch_size, args.gamma, args.interval)

    reward_list = []
    for episode_i in range(args.episode_length):
        state, info = env.reset()
        episode_reward = 0

        for step_i in range(args.step_length):
            # epsilon greedy
            epsilon = np.interp(x=episode_i*args.step_length+step_i, xp=[0, args.episode_length*args.step_length/2],
                                fp=[args.epsilon_start, args.epsilon_end])
            random_sample = random.random()
            if random_sample <= epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)

            next_state, reward, terminated, truncated, info_ = env.step(action)
            agent.memory.store_transition(state, action, next_state, reward, terminated or truncated)
            agent.update()

            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break

        print(f"Episode: {episode_i+1}, Reward: {round(episode_reward, 3)}")
        reward_list.append(episode_reward)

    # save model
    save_model(os.path.dirname(os.path.realpath(__file__)), agent.q_net.state_dict(), 'dueling_dqn_q')
    # plot reward curve
    plot_reward(reward_list, 30)
