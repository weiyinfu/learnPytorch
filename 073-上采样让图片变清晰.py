import matplotlib.pyplot as plt
import skimage.data as d
import torch as t
import torchvision as tv

img = d.astronaut()
img_tensor = tv.transforms.ToTensor()(img)
print(img.shape)
w, h, c = img.shape
print(img_tensor.shape)
bilinear = t.nn.UpsamplingBilinear2d((w * 2, h * 2))
neareast = t.nn.UpsamplingNearest2d((w * 2, h * 2))


def go(f, img_tensor):
    res = f(img_tensor.view(1, *img_tensor.shape))
    res_image = tv.transforms.ToPILImage()(res[0])
    return res_image


a = (
    ('original', img),
    ('bilinear', go(bilinear, img_tensor)),
    ('neareast', go(neareast, img_tensor)),
)
fig, axes = plt.subplots(1, len(a))
axes = axes.reshape(-1)
for (name, im), ax in zip(a, axes):
    ax.imshow(im)
    ax.set_title(name)
plt.show()
