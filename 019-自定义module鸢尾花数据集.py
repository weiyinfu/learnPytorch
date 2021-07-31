"""
nn.Module是pytorch中最重要的概念，它的意思就是一个操作
"""
from collections import Counter

import torch as t
from sklearn.datasets import load_iris


def get_data():
    x, y = load_iris(True)
    print(Counter(y))
    print(x.shape, y.shape)
    x = t.from_numpy(x)
    out = t.empty(len(y), 3)  # 3个类别
    yy = t.LongTensor(y).view(-1, 1)
    out.scatter_(dim=1, index=yy, value=1)
    # print(out)
    return x, t.from_numpy(y)


class Net(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.one = t.nn.Linear(4, 8, True)
        self.active1 = t.nn.Sigmoid()
        self.two = t.nn.Linear(8, 3, False)
        self.last = t.nn.Softmax(dim=0)

    def forward(self, input: t.Tensor):
        y = self.one(input)
        y = self.active1(y)
        y = self.two(y)
        y = self.last(y)
        return y


def train():
    x, y = get_data()
    print(type(x), x.shape, type(y), y.shape, x.dtype, y.dtype)
    x = x.float()
    model = Net()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
    loss_f = t.nn.CrossEntropyLoss()
    for i in range(100000):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_f(output, y)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            yy = t.argmax(output, dim=1)
            right = t.count_nonzero(yy == y)
            acc = right / len(y)
            print(i, acc)


train()
"""
SGD optimizer效果并不太好，需要用Adam才行
94000 tensor(0.9400)
95000 tensor(0.9400)
96000 tensor(0.9400)
97000 tensor(0.9400)
98000 tensor(0.9400)
99000 tensor(0.9400)
"""
