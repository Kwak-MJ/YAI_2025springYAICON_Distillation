import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# DataLoader를 구현한건 아닙니다
# CIFAR100은 이미 pytorch에 있어서, 쉽게 불러오는 class 작성했습니다
# train.py에서 사용합니다

# 전체 수정 금지!! (transform 수정은 train.py에서 하면 됨, 여기는 defualt만 설정함)


class LoadData:
    def __init__(self, root='./data', train=True, download=True, transform=None, batch_size=256):

        self.root = root
        self.train = train
        self.download = download
        self.batch_size = batch_size

        # add various transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[
                    0.2675, 0.2565, 0.2761])
            ])
        else:
            self.transform = transform

    def get_data(self):
        if self.train:  # return trainloader

            data = datasets.CIFAR100(
                root=self.root, train=self.train, download=self.download, transform=self.transform)

            dataloader = DataLoader(
                data, batch_size=self.batch_size, shuffle=True, num_workers=2)
        else:  # return testloader

            data = datasets.CIFAR100(
                root=self.root, train=self.train, download=self.download, transform=self.transform
            )

            dataloader = DataLoader(
                data, batch_size=self.batch_size, shuffle=False, num_workers=2
            )
        return dataloader
