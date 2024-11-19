# 核心思想
在DQN的基础上只有一个改动：DQN是target_q_net选取下一状态的最大值动作，然后算出它的Q值（这种思想会出现**高估**问题）。Double-DQN是使用q_net选择最大值的动作，然后使用target_q_net计算Q值。
# 实现
![double-dqn.png](./../images/double-dqn.png)
![dqn-game.png](./../images/dqn-game.png)