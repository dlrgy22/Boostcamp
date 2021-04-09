import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CustomLoss(nn.Module):
    def __init__(self, gamma=2., reduction='mean', classes=18, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        
        ce_loss =  nn.functional.cross_entropy(y_pred, y_true)
        
        y_true = torch.nn.functional.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        
        f1_loss = 1 - f1.mean()
        
        
        
        return f1_loss + ce_loss