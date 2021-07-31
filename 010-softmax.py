import torch as t

"""
softmax取e求平均
"""
a = t.Tensor([1, 2, 3])
y = t.softmax(a, dim=0)
print(y)
print(t.exp(a) / t.sum(t.exp(a)))
