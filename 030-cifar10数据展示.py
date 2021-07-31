from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

import mnist

"""
cifar10 
50000个训练
10000个测试
十个类别：猫、狗、鸟、青蛙、马、鹿、卡车、汽车、飞机、船
两个大类：动物6个，交通工具4个
32*32*3的图片形状

cifar10的难度比手写识别难一点
"""
ds = mnist.Cifar10.train
# data的每张图片的shape是32，32，3
print(ds.data.shape, type(ds.data))
targets = np.array([i[1] for i in ds])
images = np.array([np.array(i[0]) for i in ds])
print(Counter(targets))
# images每张图片的shape是3，32，32
print(images.shape)
img, _ = ds[0]
print(type(img))


def draw():
    c = Counter(targets)
    fig, axes = plt.subplots(len(c), 15)
    for row, (v, cnt) in zip(axes, c.items()):
        a = images[ds.targets == v]
        b = a[:len(row)]
        for img, axe in zip(b, row):
            img = np.transpose(img, [1, 2, 0], )
            axe.imshow(img)
            axe.axis('off')
    plt.show()

# draw()
