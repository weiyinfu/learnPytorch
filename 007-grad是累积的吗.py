import torch as t

x = t.rand(2)
x.requires_grad = True
y = x[0] * 3 + x[1] * 4
for i in range(10):
    y.backward(retain_graph=True)
    print(x.grad)
    # 如果不清空grad，grad就会累积.optimizer.zero_grad()就是清空梯度
    # x.grad=None
