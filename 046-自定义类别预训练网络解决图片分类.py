import torch as t
import torchvision
from torch.utils import data

import mnist

"""
预训练模型的最后一层bias=True
(fc): Linear(in_features=2048, out_features=1000, bias=True)
"""
train_data = mnist.Cifar10.train
test_data = mnist.Cifar10.test
m = t.nn.Sequential(
    torchvision.models.resnet18(True, True),
    t.nn.ReLU(True),
    t.nn.Linear(in_features=1000, out_features=10, bias=False)
)
print(m[0])


def train():
    loader = data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)
    opt = t.optim.Adam(m.parameters(), lr=1e-4)
    loss_f = t.nn.CrossEntropyLoss()

    def get_test_acc():
        right = 0
        for x, y in test_loader:
            yy = m(x)
            right += t.count_nonzero(t.argmax(yy, 1) == y)
        return right / len(test_data.data)

    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            opt.zero_grad()
            y = m(batch_x)
            l = loss_f(y, batch_y)
            l.backward()
            opt.step()
            if step % 100 == 0:
                acc = t.count_nonzero(t.argmax(y, dim=1) == batch_y) / len(batch_y)
                real_acc = get_test_acc()
                print(f"epoch={epoch} step={step} loss={l} acc={acc} real_acc={real_acc}")


train()
