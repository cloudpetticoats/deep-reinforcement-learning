import gym
import matplotlib.pyplot as plt

from agent import Agent

# Hyperparameters
MAX_EPISODE = 300
MAX_TRAJECTORY_LENGTH = 200
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4
GAMMA = 0.99
LAMBDA = 0.95
EPOCH = 10
EPS = 0.15


reward_list = []


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')  # render_mode="human"
    n_states = env.observation_space.shape[0]  # 3
    n_actions = env.action_space.shape[0]  # 1

    agent = Agent(n_states, n_actions, MAX_TRAJECTORY_LENGTH, ACTOR_LR, CRITIC_LR, GAMMA, LAMBDA, EPOCH, EPS)
    for i in range(MAX_EPISODE):
        state, info = env.reset()
        done = False
        reward_total = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent.trajectory.add(state, action, next_state, reward, done)
            state = next_state
            reward_total += reward

        reward_list.append(reward_total)

        agent.update()
        agent.trajectory.clear()
        print(f"Episode：{i + 1}, Reward：{reward_total}")

    agent.plot(agent.actor_loss, 'actor_epoch')
    agent.plot(agent.critic_loss, 'critic_epoch')

    plt.plot(range(len(reward_list)), reward_list, color='r')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
