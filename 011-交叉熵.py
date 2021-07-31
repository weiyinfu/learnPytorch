import torch as t

f = t.nn.CrossEntropyLoss()
a = t.rand(1, 3, dtype=t.float64)
print(a)
target = t.LongTensor([2])
print(f(a, target))


def cross_entropy(a, target):
    a = t.softmax(a, dim=1)
    w = t.zeros_like(a, dtype=t.float64)
    for ind, i in enumerate(target):
        w[ind, i] = 1
    return t.sum(-w * t.log(a), dim=1)


print(cross_entropy(a, target))
