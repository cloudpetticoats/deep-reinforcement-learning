import matplotlib.pyplot as plt


def plot(actor_loss, critic_loss, reward):
    # 创建子图
    fig, axs = plt.subplots(3, 3, figsize=(25, 20))

    axs[0, 0].plot(range(len(actor_loss[0])), actor_loss[0], color='purple')
    axs[0, 0].set_xlabel('step')
    axs[0, 0].set_ylabel('adversary_actor loss')

    axs[1, 0].plot(range(len(actor_loss[1])), actor_loss[1], color='purple')
    axs[1, 0].set_xlabel('step')
    axs[1, 0].set_ylabel('agent0_actor loss')

    axs[2, 0].plot(range(len(actor_loss[2])), actor_loss[2], color='purple')
    axs[2, 0].set_xlabel('step')
    axs[2, 0].set_ylabel('agent1_actor loss')

    axs[0, 1].plot(range(len(critic_loss[0])), critic_loss[0], color='purple')
    axs[0, 1].set_xlabel('step')
    axs[0, 1].set_ylabel('adversary_critic loss')

    axs[1, 1].plot(range(len(critic_loss[1])), critic_loss[1], color='purple')
    axs[1, 1].set_xlabel('step')
    axs[1, 1].set_ylabel('agent0_critic loss')

    axs[2, 1].plot(range(len(critic_loss[2])), critic_loss[2], color='purple')
    axs[2, 1].set_xlabel('step')
    axs[2, 1].set_ylabel('agent_1_critic loss')

    axs[0, 2].plot(range(len(reward)), reward, color='purple')
    axs[0, 2].set_xlabel('episode')
    axs[0, 2].set_ylabel('episode reward')

    plt.subplots_adjust(hspace=0.5)
    plt.show()