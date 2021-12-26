import torch as t

embedding = t.nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = t.autograd.Variable(t.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]))
print(embedding(input))
# 设置权重
embedding.weight.data.copy_(t.rand(10, 3))
print(embedding(input))