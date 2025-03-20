import torch
import os.path
import gym

from agent import Actor


if __name__ == '__main__':
    # load trained model
    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = current_path + '/models/sac_actor_20250320211156.pth'
    env = gym.make('Pendulum-v1', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    actor = Actor(state_dim, action_dim, max_action)
    actor.load_state_dict(torch.load(model_path))

    # test
    for episode_i in range(50):
        state, info = env.reset()
        episode_reward = 0

        for step_i in range(200):
            action, _ = actor(torch.FloatTensor(state).unsqueeze(0))
            next_state, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy()[0])
            if terminated or truncated:
                break
            state = next_state
            episode_reward += reward

        print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 3)}")

    env.close()
