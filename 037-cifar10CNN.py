import torch as t
from torch.utils import data

import mnist

ds = mnist.Cifar10
train_data = ds.train
test_data = ds.test


class Flatten(t.nn.Module):
    # 写一个平铺层
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


inchannel = 3 if ds == mnist.Cifar10 else 1
output_shape = 288 if ds == mnist.Cifar10 else 200
m = t.nn.Sequential(
    t.nn.Conv2d(in_channels=inchannel, out_channels=16, kernel_size=(3, 3), stride=(1,)),
    t.nn.ReLU(),
    t.nn.MaxPool2d(2),
    t.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3)),
    t.nn.ReLU(),
    t.nn.MaxPool2d(2),
    Flatten(),
    t.nn.Linear(in_features=output_shape, out_features=100),
    t.nn.Sigmoid(),
    t.nn.Linear(in_features=100, out_features=10)
)
model_path = "data/cifar10-cnn.pt"


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
