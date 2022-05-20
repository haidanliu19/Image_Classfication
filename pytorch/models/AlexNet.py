# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

class BaseModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        
        self.features1 = nn.Sequential(
            # input 227x227x3 -> output 96, 55, 55
            nn.Conv2d(in_channels = 3, out_channels = 96, stride = 4, kernel_size = (11, 11)),
            nn.ReLU(inplace = True),
            
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 265, stride = 1, kernel_size = (5, 5), padding=2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels = 265, out_channels = 384, stride = 1, kernel_size = (3, 3), padding=1),
            nn.ReLU(inplace = True)
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(in_channels = 384, out_channels = 256, stride = 1, kernel_size = (3, 3), padding=1),
            nn.ReLU(inplace = True)
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, stride = 1, kernel_size = (3, 3), padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, 10)
        )
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = torch.flatten(x, 1) # torch.Size([2, 9216])
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = BaseModel().to("cuda")
    from torchsummary import summary
    summary(model, input_size=(3, 227, 227))

