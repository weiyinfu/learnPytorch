import torch as t

x, y = t.rand(2, 1, 3)
dis = t.nn.PairwiseDistance(2)
print(dis(x, y))


def mydis(p, x, y):
    return t.sum((x[0] - y[0]) ** p) ** (1 / p)


print(mydis(2, x, y))
