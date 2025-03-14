import os.path
import torch
import torch.nn as nn
import gym


class Actor(nn.Module):
    def __init__(self, state_num, action_num, hidden_num=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_num, hidden_num)
        self.fc2 = nn.Linear(hidden_num, hidden_num)
        self.fc3 = nn.Linear(hidden_num, action_num)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * 2
        return x


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = current_path + '/models/ddpg_actor_20240709125035.pth'

    environment = gym.make(id='Pendulum-v1', render_mode="human")
    state_num = environment.observation_space.shape[0]
    action_num = environment.action_space.shape[0]
    actor = Actor(state_num, action_num)
    actor.load_state_dict(torch.load(model_path))

    for episode_i in range(50):
        state, info = environment.reset()
        episode_reward = 0

        for step_i in range(200):
            action = actor(torch.FloatTensor(state).unsqueeze(0)).detach().cpu().numpy()[0]
            next_state, reward, terminated, truncated, info = environment.step(action)
            state = next_state
            episode_reward += reward

        print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 3)}")

    environment.close()
