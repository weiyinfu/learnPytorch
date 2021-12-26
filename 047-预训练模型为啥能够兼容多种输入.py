import torchvision as tv
import torch as t

"""
cifar10是32*32*3的图片
mnist是28*28*1的图片
为啥预训练模型能够适配这两种？

有三个地方需要变？
1. 输入时的深度，第一个卷积层的in_channels，它的数值固定位3
2. 第一个全连接层的输入神经元个数，通过全局池化层把整个图片的尺寸变为固定尺寸（宽度和高度）
3. 最后一个全连接层输出的类别数，这是模型参数,num_classes。如果不指定num_classes，vgg默认的类别是1000，兼容mnist和cifar10的十个类别。
"""

x = tv.models.vgg16()
print(x)
"""
图像是三维数据：batch,channel,width,height
音频是二维数据：batch,channel,width

因此AdaptiveAvgPool1d，保留batch和channel不变，只把width变成一个特定宽度。  
CNN对于尺寸要求比较严格，RNN对尺寸要求不严格。那么CNN存在局限性吗？
答曰：使用AdaptiveAvgPool可以解决尺寸自适应问题
"""
a=t.rand(2,3,4)
p=t.nn.AdaptiveAvgPool1d(2)
print(p(a))
