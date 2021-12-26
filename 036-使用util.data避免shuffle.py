import torch as t
from torch.utils import data

x = data.DataLoader(t.arange(0, 100), batch_size=3, shuffle=False)
print(type(x), dir(x))

for epoch in range(3):
    print(f"epoch = {epoch} \n{'=' * 30}")
    for i in x:
        print(i)

"""
下面自定义一个数据集
"""


class RandomDataset(data.Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = t.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


rand_loader = data.DataLoader(dataset=RandomDataset(7, 3), batch_size=2, shuffle=True)
