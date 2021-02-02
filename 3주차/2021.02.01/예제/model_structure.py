import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, input_features=28*28*3, output_features=10, linear=[256], init_weight="he", init_bias="zero"):
        super(Model, self).__init__()
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.linear_list = []
        prev_dim = input_features
        for dim in linear:
            self.linear_list.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.output_layer = nn.Linear(prev_dim, output_features)
        self.init_params()


    def init_params(self):
        init_weight_method = {
            "he" : nn.init.kaiming_normal_,
            "xavier" : nn.init.xavier_normal_,
        }
        assert(
            self.init_weight in init_weight_method.keys()
        ), f"Select the weight initialization method in {list(init_weight_method.keys())}"

        init_bias_method = {
            "zero": nn.init.zeros_,
            "uniform": nn.init.uniform_
        }
        assert(
            self.init_bias in init_bias_method.keys()
        ), f"Select the bias initialization method in {list(init_bias_method.keys())}"

        for param_name, param in self.named_parameters():
            if "weight" in param_name:
                init_weight_method[self.init_weight](param)

            elif "bias" in param_name:
                init_bias_method[self.init_bias](param)
        
    def forward(self, X):
        for linear in self.linear_list:
            X = F.relu(linear(X))
        return F.softmax(self.output_layer(X))
        
if __name__ == "__main__":
    M = Model()
    x_numpy = np.random.rand(2,784)
    x_torch = torch.from_numpy(x_numpy).float().to("cpu")
    y_torch = M.forward(x_torch) # forward path
    y_numpy = y_torch.detach().cpu().numpy() # torch tensor to numpy array
    print ("x_numpy:\n",x_numpy)
    print ("x_torch:\n",x_torch)
    print ("y_torch:\n",y_torch)
    print ("y_numpy:\n",y_numpy)