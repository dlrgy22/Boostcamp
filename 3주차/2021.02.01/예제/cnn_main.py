import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

from config import Config
from model_structure import Model
from cnn_model_structure import CNNModel

SEED = 42
torch.manual_seed(SEED)

def get_config():
    parser = argparse.ArgumentParser(description="Multi-layer perceptron")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    args = parser.parse_args()

    config = Config(
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    return config


def get_data(BATCH_SIZE):
    data_set = ImageFolder(root='./notMNIST_small', transform=transforms.Compose([transforms.ToTensor(), ]))
    train_set, validation_set = train_test_split(data_set, test_size=0.1, random_state=123, shuffle=True)
    train_iter = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    validation_iter = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    return train_iter, validation_iter


def get_cnnmodel(LEARNING_RATE, device):
    model = CNNModel(convdims=[32, 64, 128], fcdims=[128]).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, loss_function, optimizer
    

def get_model_info(model):
    np.set_printoptions(precision=3)
    n_param = 0
    for p_idx,(param_name,param) in enumerate(model.named_parameters()):
        param_numpy = param.detach().cpu().numpy()
        n_param += len(param_numpy.reshape(-1))
        print (f"[{p_idx}] name:[{param_name}] shape:[{param_numpy.shape}].")
        print (f"    val:{param_numpy.reshape(-1)[:5]}")
    print (f"Total number of parameters:[{n_param}].")


def test_eval(model, test_iter, batch_size, device):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        total = 0
        correct = 0
        
        for batch_img, batch_lab in test_iter:
            X = batch_img.to(device)
            Y = batch_lab.to(device)
            y_pred = model(X)
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == Y).sum().item()
            total += batch_img.size(0)
        
        val_acc = 100 * correct / total
    model.train()

    return val_acc


def train_model(model, train_iter, test_iter, epochs, batch_size, device):
    print("Start training")
    
    model.init_params()
    model.train()
    for epoch in range(epochs):
        loss_val_sum = 0
        for batch_img, batch_lab in tqdm(train_iter):

            X = batch_img.to(device)
            Y = batch_lab.to(device)

            y_pred = model.forward(X)
            loss = loss_function(y_pred, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val_sum += loss
    
        loss_val_avg = loss_val_sum / len(train_iter)
        test_acc_val = test_eval(model, test_iter, batch_size, device)
        train_acc_val = test_eval(model, train_iter, batch_size, device)
        print(f"epoc:[{epoch+1}/{epochs} cost:[{loss_val_avg:.3f}] test_acc : {test_acc_val:.3f} train_Acc : {train_acc_val:.3f}")
    
    print("Training Done")


if __name__ == "__main__":
    config = get_config()
    print(config.device)
    train_iter, test_iter = get_data(config.BATCH_SIZE)
    model, loss_function, optimizer = get_cnnmodel(config.LEARNING_RATE, config.device)
    get_model_info(model)
    train_model(model, train_iter, test_iter, config.EPOCHS, config.BATCH_SIZE, config.device)