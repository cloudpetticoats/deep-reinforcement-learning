import torch.nn as nn
import torch
import os.path
import gym


class Q_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Q_net, self).__init__()
        self.f1 = nn.Linear(input_dim, 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        return self.f3(x)


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = current_path + '/models/dqn_q_net_20240712101605.pth'

    environment = gym.make('CartPole-v1', render_mode='human')
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.n

    q_net = Q_net(state_dim, action_dim)
    q_net.load_state_dict(torch.load(model_path))

    for episode_i in range(50):
        state, info = environment.reset()
        episode_reward = 0

        for step_i in range(100):
            action = q_net(torch.tensor(state)).max(0).indices.item()
            next_state, reward, terminated, truncated, info = environment.step(action)
            if terminated or truncated:
                break
            state = next_state
            episode_reward += reward

        print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 3)}")

    environment.close()
