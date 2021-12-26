import skimage.data as d
import torch as t
import torchvision as tv

img = tv.transforms.ToTensor()(d.astronaut())
img2 = t.nn.AdaptiveAvgPool2d((100, 100))(img)
small = tv.transforms.ToPILImage()(img2)
small.show()
