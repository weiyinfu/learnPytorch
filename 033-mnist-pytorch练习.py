import torch
import torch.nn as nn
import torch.utils.data as Data

from mnist import Mnist

train_data = Mnist.train
test_data = Mnist.test
model_file = 'data/cnn_minist.pkl'


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.output = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.output(out)
        return out


def train():
    EPOCH = 3
    BATCH_SIZE = 50
    LR = 0.001

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    # 只有在训练的时候才会自动压缩，所以这里采用手动压缩
    test_x = torch.unsqueeze(test_data.data, dim=1).float()[:2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.targets[:2000]

    cnn = CNN()
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, )
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                yy = cnn(test_x)
                accuracy = torch.count_nonzero(torch.argmax(yy, dim=1) == test_y) / len(test_y)
                print(f'Epoch: {epoch} | train loss: {loss:.4f} | test accuracy: {accuracy:.2f}')

    torch.save(cnn, model_file)
    print('finish training')


def test():
    # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.float) / 255.
    test_y = test_data.targets
    cnn = torch.load(model_file)
    yy = cnn(test_x)
    accuracy = torch.count_nonzero(torch.argmax(yy, dim=1) == test_y) / len(test_y)
    print('accuracy', accuracy)


train()
# test()
