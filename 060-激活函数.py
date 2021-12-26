import math

import matplotlib.pyplot as plt
import torch

input = torch.linspace(-2, 2, 100)


def softshrink(x, lambd=0.5):
    y = torch.zeros_like(x)
    x[y < lambd] += lambd
    x[y > lambd] -= lambd
    return x


def softsign(x):
    return x / (1 + torch.abs(x))


a = [
    ('threshold', torch.nn.functional.threshold(input, threshold=0.5, value=0.7, inplace=False)),  # input>threshold?x:value
    ('relu', torch.nn.functional.relu(input, inplace=False)),
    ('hardtanh', torch.nn.functional.hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False)),
    ('relu6', torch.nn.functional.relu6(input, inplace=False)),
    ('elu', torch.nn.functional.elu(input, alpha=1.0, inplace=False)),
    ('leaky_relu', torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)),
    # torch.nn.functional.prelu(input, weight),
    ('rrelu', torch.nn.functional.rrelu(input, lower=0.125, upper=0.3333333333333333, training=False, inplace=False)),
    ('logsigmoid', torch.nn.functional.logsigmoid(input)),
    ('hardshrink', torch.nn.functional.hardshrink(input, lambd=0.5)),
    ('tanhshrink', torch.nn.functional.tanhshrink(input)),
    ('softsign', torch.nn.functional.softsign(input)),
    ('softsign2', softsign(input)),
    ('softplus', torch.nn.functional.softplus(input, beta=1, threshold=20)),
    # softmin，softmax，log_softmax实际上并不是正经的激活函数
    ('softmin', torch.nn.functional.softmin(input, dim=0)),
    ('softmax', torch.nn.functional.softmax(input, dim=0)),
    ('log_softmax', torch.nn.functional.log_softmax(input, dim=0)),
    ('softshrink', torch.nn.functional.softshrink(input, lambd=0.5)),
    ('softshrink2', softshrink(input, lambd=0.5)),
    # torch.nn.functional.tanh(input),# use torch.tanh instead
    ('tanh', torch.tanh(input)),
    # torch.nn.functional.sigmoid is deptecated,use torch.sigmoid
    # torch.nn.functional.sigmoid(input),
    ('sigmoid', torch.sigmoid(input)),
]
rows = int(len(a) ** 0.5)
cols = math.ceil(len(a) / rows)
fig, axes = plt.subplots(rows, cols)
axes = axes.reshape(-1)[:len(a)]
for (name, y), ax in zip(a, axes):
    print(name)
    ax.set_title(name)
    ax.plot(input, y)
plt.show()
