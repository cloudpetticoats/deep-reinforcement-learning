import os.path
import gym
import argparse

from agent import Agent
from common.tools import plot_reward, save_model


"""
～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
|    Note: Noisy-DQN does not converge.                                        |
|    There are issues with the current code, and it will be improved later.    |
|    Please do not use it for now.                                             |
～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
"""


def init_parameters():
    """
    Initialize the parameters required for the algorithm.
    """
    parser = argparse.ArgumentParser(description="DQN Hyperparameters")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--episode_length", type=int, default=200, help="Total episode length")
    parser.add_argument("--step_length", type=int, default=100, help="step length for each episode")
    parser.add_argument("--epsilon_start", type=float, default=1, help="epsilon greedy explore epsilon_start")
    parser.add_argument("--epsilon_end", type=float, default=0.02, help="epsilon greedy explore epsilon_end")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for q network")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for reward")
    parser.add_argument("--interval", type=int, default=100, help="target q network update interval")
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

        agent.q_net.f1.reset_epsilon()
        agent.target_q_net.f1.reset_epsilon()
        agent.q_net.f2.reset_epsilon()
        agent.target_q_net.f2.reset_epsilon()
        agent.q_net.f3.reset_epsilon()
        agent.target_q_net.f3.reset_epsilon()
        agent.q_net.f4.reset_epsilon()
        agent.target_q_net.f4.reset_epsilon()

        for step_i in range(args.step_length):
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
    save_model(os.path.dirname(os.path.realpath(__file__)), agent.q_net.state_dict(), 'noisy_dqn_q')
    # plot reward curve
    plot_reward(reward_list, 30)
