import torch as t

"""
https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
"""
w = t.rand(3)
l = t.nn.NLLLoss(weight=w)
print("weight", w)
y = t.rand(2, 3)
print("predict", y)
y_true = t.LongTensor([0, 1])
print("loss", l(y, y_true))
print(y[:, y_true].shape, "shape=", w[y_true].shape)
print(- y[:, y_true] @ w[y_true])
