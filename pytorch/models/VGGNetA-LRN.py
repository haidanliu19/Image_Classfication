# -*- coding: utf-8 -*-
# https://pytorch.kr/hub/pytorch_vision_vgg/
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
import torch.nn as nn
import torch

import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

# 11 layer
class BaseModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        
        self.features1 = nn.Sequential(
            # input 224x224x3
            nn.Conv2d(in_channels = 3, out_channels = 64, stride = 1, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, stride = 1, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.features34 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, stride = 1, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 256, stride = 1, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.features56 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, stride = 1, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        
        self.features78 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, stride = 1, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace = True),
        )
        self.pool = nn.MaxPool2d(2, 2) # maxpooling
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
        )
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features34(x)
        x = self.features56(x)
        x = self.features78(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = BaseModel().to("cuda")
    print(model)
    from torchsummary import summary
    summary(model, input_size=(3, 224, 224))
        # filter that only require gradient descent
        
    filtered_parameters = []
    params_num = []    
    for p in filter(lambda p : p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Tranable params : ')
    print(sum(params_num))
