import torch
import os.path
import gym

from agent import Q_net


if __name__ == '__main__':
    # load trained model
    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = current_path + '/models/dqn_q_20240712092431.pth'
    env = gym.make('CartPole-v1', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_net = Q_net(state_dim, action_dim)
    q_net.load_state_dict(torch.load(model_path))

    # test
    for episode_i in range(50):
        state, info = env.reset()
        episode_reward = 0

        for step_i in range(100):
            action = q_net(torch.tensor(state)).max(0).indices.item()
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            state = next_state
            episode_reward += reward

        print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 3)}")

    env.close()
