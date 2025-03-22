import os.path
import gym
import argparse

from agent import Agent
from common.tools import plot_reward, save_model


def init_parameters():
    """
    Initialize the parameters required for the algorithm.
    """
    parser = argparse.ArgumentParser(description="Noisy-DQN Hyperparameters")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--episode_length", type=int, default=200, help="Maximum number of steps per episode")
    parser.add_argument("--step_length", type=int, default=100, help="Number of steps per training iteration")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training the Q-network")
    parser.add_argument("--buffer_size", type=int, default=10000, help="The size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of experiences sampled from the buffer for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
    parser.add_argument("--interval", type=int, default=10, help="Number of steps between target Q-network updates")
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
