import argparse
import gym
import os

from agent import SAC_Agent
from common.tools import plot_reward, save_model


def init_parameters():
    """
    Initialize the parameters required for the algorithm.
    """
    parser = argparse.ArgumentParser(description="SAC Hyperparameters")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--episode_length", type=int, default=200, help="Total episode length")
    parser.add_argument("--step_length", type=int, default=200, help="Step length for each episode")
    parser.add_argument("--actor_lr", type=float, default=1e-3, help="Learning rate for actor network")
    parser.add_argument("--critic_lr", type=float, default=3e-3, help="Learning rate for critic network")
    parser.add_argument("--entropy_lr", type=float, default=3e-4, help="Learning rate for entropy tuning")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor for reward")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter for target networks")
    arg = parser.parse_args()
    print(arg)
    return arg


if __name__ == '__main__':
    args = init_parameters()
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent = SAC_Agent(state_dim, action_dim, max_action, args.actor_lr, args.critic_lr, args.entropy_lr, args.buffer_size, args.batch_size, args.gamma, args.tau)

    reward_list = []
    for episode_i in range(args.episode_length):
        state, info = env.reset()
        episode_reward = 0

        for step_i in range(args.step_length):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.replay_buffer.store_transition(state, action, next_state, reward, terminated or truncated)
            state = next_state

            episode_reward += reward
            agent.update()
            if terminated or truncated:
                break

        print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 3)}")
        reward_list.append(episode_reward)
    env.close()

    # save model
    save_model(os.path.dirname(os.path.realpath(__file__)), agent.actor.state_dict(), 'sac_actor')
    # plot reward curve
    plot_reward(reward_list, 30)
