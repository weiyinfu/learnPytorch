import torch as t

"""
t.nn.Parameter()是一种特殊的Tensor类型，它一定是有梯度的

也可以使用register_parameter进行注册

t.nn.Module这个类型之所以能够监听t.nn.Parameter是因为它监听了__setattr__()，向当前object的属性赋值的时候，就会发生注册
"""


class X(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = t.nn.Parameter(t.rand(3, 3))
        w = t.nn.Parameter(t.rand(3, 3))
        self.register_parameter("bias", None)
        self.register_parameter("bias2", t.nn.Parameter(t.rand(2, 3)))


x = X()
for i in x.parameters():
    print('param', type(i), i)
