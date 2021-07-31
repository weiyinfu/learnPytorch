import os
from collections import Counter
from os import path
from os.path import *

import matplotlib.pyplot as plt
import torchvision

cur = path.abspath(path.dirname(__file__))
data_dir = join(cur, "data")
mnist_dir = join(data_dir, 'mnist')
cifar10_dir = join(data_dir, "cifar10")
if not path.exists(data_dir):
    os.makedirs(data_dir)


def should_download(folder: str):
    return not os.path.exists(folder)


class Mnist:
    train = torchvision.datasets.MNIST(
        root=mnist_dir,
        train=True,  # 需要训练数据还是测试数据
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,  # target的形变
        download=should_download(mnist_dir),
    )
    test = torchvision.datasets.MNIST(
        root=mnist_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=should_download(mnist_dir),
    )


class Cifar10:
    train = torchvision.datasets.CIFAR10(root=cifar10_dir, download=should_download(cifar10_dir),
                                         transform=torchvision.transforms.ToTensor(), )
    test = torchvision.datasets.CIFAR10(root=cifar10_dir, download=should_download(cifar10_dir), train=False, transform=torchvision.transforms.ToTensor(), )


def show_dataset_images(ds):
    print(ds)
    c = Counter(ds.targets.numpy())
    print('类别分布', c)
    fig, axes = plt.subplots(len(c), 15)
    for row, (v, cnt) in zip(axes, c.items()):
        a = ds.data[ds.targets == v]
        b = a[:len(row)]
        for img, axe in zip(b, row):
            axe.imshow(img)
            axe.axis('off')
    plt.show()
