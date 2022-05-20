# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

# Input size : 32X32 1채널
# C1 : featuremap  28 * 28  kernel 6
# S2 : subsampling 14 * 14 kernel 6
# C3 : featuremap 10*10 kernel 16
# S4 : subsampling 5*5 kernel 16개
# C1,C3,C5 : 컨볼루션 층( 5x5크기의 피처 맵)
# C5 : 120
# F6 : 84
# OUPUT : 10

class BaseModel(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(BaseModel, self).__init__()
        self.C1 = nn.Conv2d(in_channels = 3, out_channels = 6, stride = 1, kernel_size = (5, 5))
        self.S2 = nn.MaxPool2d((2,2))
        self.C3 = nn.Conv2d(in_channels = 6, out_channels = 16, stride = 1, kernel_size = (5, 5))
        self.S4 = nn.MaxPool2d((2,2))
        self.C5 = nn.Conv2d(in_channels = 16, out_channels = 120, stride = 1, kernel_size = (5, 5))
        self.F6 = nn.Linear(120, 84)
        self.F7 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.C1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = self.C5(x)
        
        #x = x.view(-1, 120)
        x = torch.flatten(x, start_dim = 1)
        x = self.F6(x)
        out = self.F7(x)
        return out
    
if __name__ == "__main__":
    model = BaseModel().to('cuda')
    from torchsummary import summary
    summary(model, input_size=(3, 32, 32))

