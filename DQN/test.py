from collections import deque, namedtuple
import random
import torch.nn as nn
import torch
import torch.optim as optim

experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
if __name__ == '__main__':
    deque = deque([], maxlen=100)
    # deque.append(experience(1, 'a', 1.1, 0.1))
    # deque.append(experience(2, 'b', 1.2, 0.2))
    # deque.append(experience(3, 'c', 1.3, 0.3))
    deque.append((torch.tensor((1, 11)).unsqueeze(0), 'a', 1.1, 0.1))
    deque.append((torch.tensor((6, 22)).unsqueeze(0), 'b', 1.2, 0.2))
    deque.append((torch.tensor((7, 33)).unsqueeze(0), 'c', 1.3, 0.3))

    # sample = random.sample(deque, 3)
    # print(sample)
    # r = experience(*zip(*sample))
    #
    # print(torch.tensor(list(map(lambda i: i is not None, r.next_state)), dtype=torch.bool))
    #
    # print(r.next_state)

    # print(torch.tensor(list(r.next_state)))
    # print('------')
    # print(torch.cat((1, 2, 3)))

    sample = random.sample(deque, 3)
    a, b, c, d = zip(*sample)
    print(a)
    print(torch.cat(a))
    print(torch.cat(a).max(1))

