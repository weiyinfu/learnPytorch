import torch as t

"""
两个数字齐头并进
"""
x = t.rand(2)
x.requires_grad = True
while 1:
    y = (x[0] - x[1]) ** 2
    print(f"y={y}, x={x.data}")
    if y < 1e-6:
        break
    y.backward()
    x.data.sub_(x.grad * 1e-2)
    print(f"grad={x.grad}")
print(x.data)
