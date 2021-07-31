import torch as t
"""
MSELoss就是Mean Square Error
"""
x = t.rand((2, 3))
print(x.data)
f = t.nn.MSELoss()
l = f(x[0], x[1])
print(l)
print(t.linalg.norm(x[0] - x[1]) ** 2 / len(x[0]))
