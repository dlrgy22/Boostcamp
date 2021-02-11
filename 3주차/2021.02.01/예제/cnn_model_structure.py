import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

class CNNModel(nn.Module):
    def __init__(self, input_features=[3, 28, 28], output_features=10, convdims = [32, 64], 
                fcdims = [128], ksize=3, init_weight="he", init_bias="zero", USE_BATCHNORM=False):
        super(CNNModel, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.convdims = convdims
        self.fcdims = fcdims
        self.ksize = ksize
        self.USE_BATCHNORM=USE_BATCHNORM
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.layers = []
        prev_cdim = self.input_features[0]
        for cdim in self.convdims:
            self.layers.append(
                nn.Conv2d(
                    in_channels=prev_cdim, 
                    out_channels=prev_cdim // 3,
                    kernel_size=1,
                    stride=1,
                )
            )


            self.layers.append(
                nn.Conv2d(
                    in_channels=prev_cdim // 3, 
                    out_channels=cdim,
                    kernel_size=self.ksize,
                    stride=1,
                    padding=self.ksize//2
                )
            )

            if USE_BATCHNORM:
                self.layers.append(nn.BatchNorm2d(cdim))
            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            #self.layers.append(nn.Dropout2d(p=0.7))
            prev_cdim = cdim

            check_first = False

        self.layers.append(nn.Flatten())
        prev_fcdim = prev_cdim*(self.input_features[1]//(2**len(self.convdims)))*(self.input_features[2]//(2**len(self.convdims)))
        for fcdim in self.fcdims:
            self.layers.append(nn.Linear(prev_fcdim, fcdim, bias=True))
            self.layers.append(nn.ReLU(True))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            prev_fcdim = fcdim
        self.layers.append(nn.Linear(prev_fcdim, self.output_features, bias=True))
        self.layers.append(nn.Softmax())

        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),l_idx)
            self.net.add_module(layer_name, layer)
        
        self.init_params()

            
    def init_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d): # init BN
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, X):
        return self.net(X)
        
if __name__ == "__main__":
    M = CNNModel()
    x_numpy = np.random.rand(2, 3, 28, 28)
    x_torch = torch.from_numpy(x_numpy).float().to("cpu")
    y_torch = M.forward(x_torch) # forward path
    y_numpy = y_torch.detach().cpu().numpy() # torch tensor to numpy array
    print ("x_numpy:\n",x_numpy)
    print ("x_torch:\n",x_torch)
    print ("y_torch:\n",y_torch)
    print ("y_numpy:\n",y_numpy)