import math

import matplotlib.pyplot as plt
import numpy as np
import skimage.data as d
import torchvision as tv

plt.rcParams['font.sans-serif'] = ['Consolas']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# Compose把多种变换组合起来
trans = tv.transforms.Compose([
    tv.transforms.CenterCrop(20),
    tv.transforms.RandomVerticalFlip(),
    tv.transforms.ToTensor(),
])


def one(x):
    return x + 1


def two(x):
    return x + 2


"""
Compose的本质就是把多个执行器串联起来
"""
trans2 = tv.transforms.Compose([
    one,
    two,
])

print(trans2(4))  # 输出7
img = d.astronaut()
img2 = tv.transforms.ToTensor()(img)


def get_image(op):
    center_crop = op(img2)
    return np.array(tv.transforms.ToPILImage()(center_crop))


op_list = [
    ('Original', lambda x: img),
    ('CenterCrop', tv.transforms.CenterCrop(img.shape[1] // 2)),
    ('RandomCrop', tv.transforms.RandomCrop(img.shape[1] // 2)),
    ('Pad', tv.transforms.Pad(20)),
    ('HorizonFlip', tv.transforms.RandomHorizontalFlip(1)),
    ('VerticalFlip', tv.transforms.RandomVerticalFlip(1)),

]
img_count = len(op_list)
rows = int(img_count ** 0.5)
cols = math.ceil(img_count / rows)

fig, axes = plt.subplots(rows, cols)
axes = axes.reshape(-1)[:img_count]
for ax, (op_name, op) in zip(axes, op_list):
    img = get_image(op)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(op_name)
plt.show()
