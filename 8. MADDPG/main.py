import torch
import numpy as np
import random
import argparse

from pettingzoo.mpe import simple_adversary_v3
from agent import MADDPG
from utils import plot


"""
～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
|    Note: MADDPG does not converge.                                           |
|    There are issues with the MADDPG code, and it will be improved later.     |
|    Please do not use it for now.                                             |
～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
"""


def init_parameters():
    """
    Initialize the parameters required for the algorithm.
    """
    parser = argparse.ArgumentParser(description="MA-DDPG Hyperparameters")
    parser.add_argument("--episode_length", type=int, default=1000, help="Total episode length")
    parser.add_argument("--step_length", type=int, default=25, help="Step length for each episode")
    parser.add_argument("--epsilon_start", type=float, default=1, help="Epsilon greedy explore epsilon_start")
    parser.add_argument("--epsilon_end", type=float, default=0.02, help="Epsilon greedy explore epsilon_end")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Learning rate for actor network")
    parser.add_argument("--critic_lr", type=float, default=1e-4, help="Learning rate for critic network")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.96, help="Discount factor for reward")
    parser.add_argument("--interval", type=int, default=100, help="Target network update interval")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter for target networks")
    arg = parser.parse_args()
    print(arg)
    return arg


if __name__ == '__main__':
    args = init_parameters()
    env = simple_adversary_v3.parallel_env(continuous_actions=True)
    env.reset()
    agent_num = env.num_agents
    obs_dim = []  # The observation space dimension of each agent
    action_dim = []  # The action space dimension of each agent
    reward_history = []
    agent_name_list = []

    for agent_idx, agent_name in enumerate(env.agents):
        obs_dim.append(env.observation_space(agent_name).shape[0])
        action_dim.append(env.action_space(agent_name).shape[0])
        agent_name_list.append(agent_name)

    maddpg = MADDPG(actor_input_dims=obs_dim, actor_output_dim=action_dim, critic_input_dim=sum(obs_dim)+sum(action_dim),
                    agent_num=agent_num, lr_actor=args.actor_lr, lr_critic=args.critic_lr, replay_buffer_capacity=args.buffer_size,
                    batch_size=args.batch_size, gamma=args.gamma, tau=args.tau)

    for episode_i in range(args.episode_length):
        observations, infos = env.reset()
        episode_reward = 0

        for step_i in range(args.step_length):
            actions = {}
            for agent_idx, agent_name in enumerate(env.agents):
                # epsilon greedy
                epsilon = np.interp(x=episode_i * args.step_length + step_i, xp=[0, args.episode_length * args.step_length / 2],
                                    fp=[args.epsilon_start, args.epsilon_end])
                random_sample = random.random()
                if random_sample <= epsilon:
                    actions = {agent_name: env.action_space(agent_name).sample() for agent_idx, agent_name in enumerate(env.agents)}
                else:
                    actions[agent_name] = maddpg.choose_action(observations[agent_name], agent_idx)

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            combine_observations = []
            combine_next_observations = []
            combine_actions = []
            for name_i in agent_name_list:
                combine_observations.append(observations[name_i])
                combine_next_observations.append(next_observations[name_i])
                combine_actions.append(actions[name_i])

            for agent_idx, agent_name in enumerate(env.agents):
                maddpg.replay_buffers[agent_idx].add(observations[agent_name], combine_observations, combine_actions,
                                                     rewards[agent_name], next_observations[agent_name],
                                                     combine_next_observations, terminations[agent_name])

            maddpg.update((episode_i*args.step_length+step_i)/args.interval == 0)

            episode_reward += sum(rewards.values())
            observations = next_observations

            if all(terminations.values()):
                break

        reward_history.append(episode_reward)
        if episode_i%20 == 0:
            print(f"Episode {episode_i}/{args.episode_length} | Reward: {round(episode_reward, 3)}")

    # save model
    for agent_idx, agent_name in enumerate(env.agents):
        torch.save(maddpg.actors[agent_idx], f"actor_{agent_name}.pth")

    plot(maddpg.loss_actors, maddpg.loss_critics, reward_history)

    env.close()
