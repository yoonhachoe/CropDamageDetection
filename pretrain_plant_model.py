import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=500):
        super(ResNet, self).__init__()
        self.device = torch.device('cuda')
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.resnet.fc = self.resnet.fc.requires_grad_(True)

    def forward(self, image):
        image = image.to(self.device)
        pred = self.resnet(image)

        return pred