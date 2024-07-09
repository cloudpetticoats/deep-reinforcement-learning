# _1.Introduction_
Code implementation of deep reinforcement learning

# _2.Reference_
* Excellent Weblog：https://zhuanlan.zhihu.com/p/342919579
* Excellent Web：https://www.deeprlhub.com/

# _3.Tips_
* _Add a baseline:_ Make the total reward when updating an actor have a positive or negative number. And not always positive.(因为随机sample样本训练，可能抽到不好的action去训练，并且这个action的reward又是正数，导致这个action概率增大~)
![baseline.png](images/baseline.png)
* _Assign Suitable Credit(就是在baseline的基础上再加一个衰减因子gamma作为后面梯度的权重系数):_ A `gamma` has been added, which means that the farther away from the current state of the action is made, the smaller the weight of the reward to the current one.
`b` is generated through a network and is somewhat complex.
The `Advantage Function` is the critic of Actor-Critic.
![credit.png](images/credit.png)

# _4.The algorithm included in this project_
* DDPG
