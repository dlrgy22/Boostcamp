from torch import nn
import timm
import torch

class Model(nn.Module):
    def __init__(self, model_name, n_classes, pretrained=False, dropRate=0.2):
        super().__init__()        
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=n_classes, drop_rate=dropRate)
        

    def forward(self, x):
        x = self.model(x)
        return x