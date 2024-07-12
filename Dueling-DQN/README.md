# 核心思想
+ 在DQN的基础上值改变的网络的结构，输出位置是由两个结果加起来的。并且需要给输出矢量那个部分加上个约束，例如令矢量里面的元素之和始终等于0的normalize(下图上半部分是DQN网络架构，下半部分是Dueling-DQN网络架构)
![dueling-dqn.png](./../images/dueling-dqn.png)
+ 具体细节
![dueling-dqn-1.png](./../images/dueling-dqn-1.png)
# 实现
![dueling-dqn-res.png](./../images/dueling-dqn-res.png)
![dqn-game.png](./../images/dqn-game.png)