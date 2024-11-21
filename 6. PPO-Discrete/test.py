import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.f1 = nn.Linear(input_dim, 64)
        self.f2 = nn.Linear(64, 64)
        self.action_probs = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return F.softmax(self.action_probs(x), dim=-1)  # 输出动作概率


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = current_path + '/models/ppo_actor_model.pth'

    environment = gym.make('CartPole-v0', render_mode="human")
    n_states = environment.observation_space.shape[0]  # 4
    n_actions = environment.action_space.n  # 2
    actor = PolicyNet(n_states, n_actions)
    actor.load_state_dict(torch.load(model_path))

    for episode_i in range(20):
        state, info = environment.reset()
        episode_reward = 0

        for step_i in range(200):
            with torch.no_grad():
                action_probs = actor(torch.tensor(state).unsqueeze(0))
                distribution = torch.distributions.Categorical(probs=action_probs)
                action = distribution.sample().item()
            next_state, reward, terminated, truncated, info = environment.step(action)
            done = terminated | truncated
            state = next_state
            episode_reward += reward
            if done:
                break

        print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 3)}")

    environment.close()
