import numpy as np
import torch
import torch.utils.data as d
import torchvision

import mnist

"""
dataset的transform和target_transform仅仅在通过Dataset的下标访问的时候才会临时转换

它的data，target已经是Tensor类型的对象了
"""


def test1():
    def t(a):
        print("transform", a)
        return a

    """
    transform和target_transform都是在get具体item的时候调用
    即便不指定transform，data的data和targets字段也已经是Tensor类型的数据了
    """
    x = torchvision.datasets.MNIST(mnist.mnist_dir, transform=t, target_transform=t)
    print("============")
    print(len(x))
    print(x[0])
    print(x[0])
    print(type(x.data))


def test2():
    """
    使用dataLoader的方式访问数据集的时候，dataLoader把数据集当成一个list
    如果加载Mnist数据集的时候指定transform函数，则调用transform的次数会非常多，中间并不会缓存transform的结果。
    :return:
    """
    transform_count = 0

    def t(a):
        nonlocal transform_count
        transform_count += 1
        # print('transform', a)
        aa = np.array(a)
        bb = torch.tensor(aa)
        return bb

    x = torchvision.datasets.MNIST(mnist.mnist_dir, transform=t)
    for epoch in range(3):
        print(f"epoch {epoch} {'=' * 20}")
        for batch_x, batch_y in d.DataLoader(x, batch_size=30000):
            print(type(batch_x), type(batch_y), batch_x.shape, batch_y.shape)
    print(transform_count)


def test3():
    for i in d.DataLoader(torch.rand(10, 2, 3), batch_size=4):
        print(len(i), type(i), i.shape)


test3()
