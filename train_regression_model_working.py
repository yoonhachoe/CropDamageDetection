import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

class ResNet(nn.Module):
    def __init__(self, out_size=1):
        super(ResNet, self).__init__()
        self.device = torch.device('cuda')
        #ckpt = torch.load("./pretrained_ResNet34_008epochs.pth") # resnet34
        ckpt = torch.load("./pretrained_ResNet50_003epochs.pth")['model_state_dict'] # resnet50
        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
            name = k.replace("resnet.", "")
            new_state_dict[name] = v
        new_state_dict['fc.weight'] = torch.zeros([1000, 2048]) # resnet50, comment out when using resnet34
        new_state_dict['fc.bias'] = torch.zeros([1000]) # resnet50, comment out when using resnet34
        self.resnet = models.resnet50(pretrained=True) # choose proper resnet model (resnet34/50)
        self.resnet.load_state_dict(new_state_dict)
        for param in self.resnet.parameters():
            param.requires_grad = True
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_size)
        self.resnet.fc = self.resnet.fc.requires_grad_(True)

    def forward(self, image):
        image = image.to(self.device)
        pred = self.resnet(image)
        pred = pred.view(pred.shape[0], )  # reshape from (N,1) to (N,) to avoid mismatches in the loss function

        return pred