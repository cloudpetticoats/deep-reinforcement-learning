import os.path
import torch
import gym

from agent import PolicyNet


if __name__ == '__main__':
    # load trained model
    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = current_path + '/models/ppo_actor_20250320230032.pth'
    env = gym.make('CartPole-v0', render_mode="human")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    actor = PolicyNet(n_states, n_actions)
    actor.load_state_dict(torch.load(model_path))

    # test
    for episode_i in range(50):
        state, info = env.reset()
        episode_reward = 0

        for step_i in range(200):
            with torch.no_grad():
                action_probs = actor(torch.tensor(state).unsqueeze(0))
                distribution = torch.distributions.Categorical(probs=action_probs)
                action = distribution.sample().item()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            state = next_state
            episode_reward += reward
            if done:
                break

        print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 3)}")

    env.close()
