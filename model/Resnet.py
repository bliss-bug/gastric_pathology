import torch
import torch.nn as nn
from torchvision.models import resnet50


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        model = resnet50()
        model.load_state_dict(torch.load('checkpoints/resnet50-19c8e357.pth'))
        self.model = list(model.children())[:-3]
        self.features = nn.Sequential(*self.model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    

if __name__ == '__main__':
    m = Resnet50()
    a = torch.rand(2,3,224,224)
    print(m(a).shape)