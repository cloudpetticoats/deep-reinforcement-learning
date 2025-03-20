import numpy as np
import matplotlib.pyplot as plt
import time
import torch


def moving_average(data, window_size=20):
    """
    Smooth the data using a sliding window
    """
    averaged_data = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window_data = data[start_idx:i + 1]
        avg = np.mean(window_data)
        averaged_data.append(avg)
    return np.array(averaged_data)


def compute_std(data, window_size=20):
    """
    Calculate the sliding window standard deviation
    """
    stds = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window_data = data[start_idx:i + 1]
        std_value = np.std(window_data)
        stds.append(std_value)
    return np.array(stds)


def plot_reward(reward_list, window_size=20):
    """
    Plot the reward curve.
    """
    moving_reward_list = moving_average(reward_list, window_size)
    std_reward_list = compute_std(reward_list, window_size)

    plt.plot(range(len(moving_reward_list)), moving_reward_list, color='b')
    plt.fill_between(range(len(moving_reward_list)), moving_reward_list - std_reward_list, moving_reward_list + std_reward_list,
                    color='b',
                    alpha=0.2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


def save_model(current_path, model, model_name):
    """
    Save the trained model
    """
    model_path = current_path + '/models/'
    timestamp = time.strftime("%Y%m%d%H%M%S")
    torch.save(model, model_path + f"{model_name}_{timestamp}.pth")
