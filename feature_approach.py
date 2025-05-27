import torchvision.models as models


resnet18 = models.resnet18(weights=None)

# 한 번씩 수행해보면 알 수 있습니다
# print(resnet18.__dict__)
# print(resnet18.children())

modules = list(resnet18.children())
print(modules)
