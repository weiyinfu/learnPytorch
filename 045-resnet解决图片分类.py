import torchvision

m = torchvision.models.resnet50(True, True)
print(type(m))