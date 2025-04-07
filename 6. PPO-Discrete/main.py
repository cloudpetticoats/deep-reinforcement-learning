import gym
import os
import argparse

from agent import Agent
from common.tools import plot_reward, save_model


def init_parameters():
    """
    Initialize the parameters required for the algorithm.
    """
    parser = argparse.ArgumentParser(description="PPO-Discrete Hyperparameters")
    parser.add_argument("--env_name", type=str, default="CartPole-v0", help="Environment name")
    parser.add_argument("--episode_length", type=int, default=1000, help="Maximum episodes")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Learning rate for actor network")
    parser.add_argument("--critic_lr", type=float, default=1e-4, help="Learning rate for critic network")
    parser.add_argument("--trajectory_length", type=int, default=200, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor for rewards")
    parser.add_argument("--Lambda", type=float, default=0.95, help="Decay factor in GAE (Generalized Advantage Estimation)")
    parser.add_argument("--epoch", type=int, default=25, help="Training times per trajectory")
    parser.add_argument("--eps", type=float, default=0.2, help="Clipping range")
    arg = parser.parse_args()
    print(arg)
    return arg


if __name__ == '__main__':
    args = init_parameters()
    env = gym.make(args.env_name)
    states_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.n
    agent = Agent(states_dim, actions_dim, args.trajectory_length, args.actor_lr, args.critic_lr, args.gamma, args.Lambda, args.epoch, args.eps)

    reward_list = []
    for i in range(args.episode_length):
        state, info = env.reset()
        done = False
        reward_total = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.trajectory.add(state, action, next_state, reward, done)
            state = next_state
            reward_total += reward

        agent.update()
        agent.trajectory.clear()

        reward_list.append(reward_total)
        print(f"Episode：{i + 1}, Reward：{round(reward_total, 3)}")

    # save model
    save_model(os.path.dirname(os.path.realpath(__file__)), agent.actor.state_dict(), 'ppo_actor')
    # plot reward curve
    plot_reward(reward_list, 30)
