from agent import PolicyNet

import os.path
import torch
import gym


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = current_path + '/models/ppo_actor_net.pth'

    environment = gym.make(id='Pendulum-v1', render_mode="human")
    state_num = environment.observation_space.shape[0]
    action_num = environment.action_space.shape[0]
    max_action = environment.action_space.high[0]
    actor = PolicyNet(state_num, action_num, max_action)
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
