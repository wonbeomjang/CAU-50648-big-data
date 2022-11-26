import torch.nn as nn
from torchvision.models.resnet import resnet50, ResNet50_Weights


class PogModel(nn.Module):
    def __init__(self):
        super(PogModel, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(512 * 4, 768)
        self.classifier = nn.Linear(768, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


