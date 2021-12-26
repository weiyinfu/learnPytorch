"""
卷积者：连续区域内若干个数值聚合也

pytorch的卷积层一律都是valid模式，无法指定same，full模式
"""
import torch as t

x = t.randint(0, 10, (2, 1, 7,), dtype=t.float)
y = t.nn.Conv1d(1, 1, (3,))(x)
print(y.shape)
