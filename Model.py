import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import torch.nn as nn
import torch


def resnet18():
    resnet18 = models.resnet18(
        weights=None)

    for param in resnet18.parameters():
        param.requires_grad = True

    resnet18.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet18.maxpool = nn.Identity()
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 100)

    return resnet18


def resnet50():
    resnet50 = models.resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V2)

    for param in resnet50.parameters():
        param.requires_grad = True

    resnet50.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet50.maxpool = nn.Identity()
    resnet50.fc = nn.Linear(resnet50.fc.in_features, 100)

    return resnet50

# feature 추출 위한 Class


class modelWithFeat(nn.Module):
    def __init__(self, base_model, isStudent=False):
        super().__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = list(base_model.children())[-2]
        self.fc = list(base_model.children())[-1]

        self.isStudent = isStudent
        self.proj = nn.Conv2d(512, 2048, 1)

    def forward(self, x):
        feat = self.base(x)
        pooled = self.avgpool(feat)
        out = self.fc(pooled.view(x.shape[0], -1))

        if self.isStudent:
            feat = self.proj(feat)

        return out, feat
###


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, feat_s, feat_t):
        # Flatten: (B, C, H, W) → (B, C)
        if len(feat_s.shape) > 2:
            feat_s = torch.flatten(feat_s, start_dim=1)
            feat_t = torch.flatten(feat_t, start_dim=1)
        target = torch.ones(feat_s.shape[0]).to(
            feat_s.device)  # +1 for similarity
        return self.cos(feat_s, feat_t, target)
