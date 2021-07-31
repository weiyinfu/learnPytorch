import torch as t
from torch.utils import data

x = data.DataLoader(t.arange(0, 100), batch_size=3, shuffle=False)
print(type(x), dir(x))

for epoch in range(3):
    print(f"epoch = {epoch} \n{'=' * 30}")
    for i in x:
        print(i)
