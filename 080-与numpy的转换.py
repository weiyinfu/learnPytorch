import numpy as np
import torch as t

x = np.random.rand(2, 2)
y = t.from_numpy(x)
print(y)
y[0, 0] = 100
print(x, y)  # x和y同时发生变化，说明torch的数据是完全复用numpy的
print("=========")
# 要想让tensor重新复制一份，则直接使用t.tensor即可
yy = t.tensor(x)
yy[0, 0] = 200
print(x,yy)
print("=======tensor转numpy")
print(type(yy.numpy()))
