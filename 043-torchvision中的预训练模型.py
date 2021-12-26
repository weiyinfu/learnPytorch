import torchvision as t
from tqdm import tqdm
"""
探究预训练模型的模型结构

把模型结构保存到文本文件
"""
# resnet系列
models = [
    t.models.resnet50,
    t.models.resnet18,
    t.models.resnet34,
    t.models.resnet101,
    t.models.resnet101,
    t.models.resnet152,
    t.models.resnext50_32x4d,
    t.models.resnext101_32x8d,
    t.models.wide_resnet50_2,
    t.models.wide_resnet101_2,
    # alexnet
    t.models.alexnet,
    t.models.googlenet,
    # vgg系列
    t.models.vgg11,
    t.models.vgg11_bn,
    t.models.vgg13,
    t.models.vgg13_bn,
    t.models.vgg16,
    t.models.vgg16_bn,
    t.models.vgg19,
    t.models.vgg19_bn,
    # densenet
    t.models.densenet121,
    t.models.densenet161,
    t.models.densenet169,
    t.models.densenet201,

    # inception-v3
    t.models.inception_v3,
]
with open('pretrained-models.txt', 'w') as f:
    for i in tqdm(models):
        m = i()
        f.write(str(m))
        f.write('\n')
