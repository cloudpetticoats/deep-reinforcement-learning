import gym
import os.path
import argparse

from agent import Agent
from normalization import Normalization, RewardScaling
from common.tools import plot_reward, save_model


def init_parameters():
    """
    Initialize the parameters required for the algorithm.
    """
    parser = argparse.ArgumentParser(description="PPO-Continue Hyperparameters")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--episode_length", type=int, default=1000, help="Total episode length")
    parser.add_argument("--actor_lr", type=float, default=2e-4, help="Learning rate for actor network")
    parser.add_argument("--critic_lr", type=float, default=2e-3, help="Learning rate for critic network")
    parser.add_argument("--trajectory_length", type=int, default=250, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor for reward")
    parser.add_argument("--Lambda", type=float, default=0.9, help="Lambda for PPO")
    parser.add_argument("--epoch", type=int, default=20, help="Train epoch times for each trajectory")
    parser.add_argument("--eps", type=float, default=0.2, help="Clip")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy_coef")
    arg = parser.parse_args()
    print(arg)
    return arg


if __name__ == '__main__':
    args = init_parameters()
    env = gym.make(args.env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    # state_norm = Normalization(shape=n_states)
    scaling_reward = RewardScaling(shape=1, gamma=args.gamma)

    agent = Agent(n_states, n_actions, args.trajectory_length, args.actor_lr, args.critic_lr,
                  args.gamma, args.Lambda, args.epoch, args.eps, max_action, args.entropy_coef)
    reward_list = []
    for i in range(args.episode_length):
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

        agent.update()
        agent.trajectory.clear()

        reward_list.append(reward_total)
        print(f"Episode：{i + 1}, Reward：{reward_total}")

    # save model
    save_model(os.path.dirname(os.path.realpath(__file__)), agent.actor.state_dict(), 'ppo_policy')
    # plot reward curve
    plot_reward(reward_list, 30)
