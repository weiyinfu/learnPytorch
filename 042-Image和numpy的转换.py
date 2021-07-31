import numpy as np
import skimage.data as d
import torchvision.transforms as t
from PIL import Image

"""
从skimage中加载一张图片，加载得到的结果是一个ndarray
"""
x = d.astronaut()
print(type(x), x.shape)
"""
ndarray转Image
"""
img = Image.fromarray(x)
print(type(img))
img.show()
"""
PIL Image转numpy数组，直接转换即可
"""
a = np.array(img)
print(a.shape)
"""
pytorch中的ToTensor把PIL Image或者ndarray转成Tensor，pytorch中的图片是channel，width，height而不是width，height，channel。

ToTensor
"""
to_tensor = t.ToTensor()
tensor = to_tensor(img)
print(tensor.shape, type(tensor))
print("tensor的data", tensor)  # tensor的data已经变成了0到1之间的数字
"""
tensor Image转PIL Image
"""
to_image = t.ToPILImage()
img2 = to_image(tensor)
print(type(img2))
# img2.show()
