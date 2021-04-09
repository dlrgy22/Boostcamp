import argparse
import time
import warnings

import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from config import Config
from timm_model import Model

from data_set import Image_Dataset_
from albumentations import *

from albumentations.pytorch.transforms import ToTensorV2

def get_test_transforms():
    return Compose([
                CenterCrop(380, 380),
                Normalize(
                    mean=[0.5610, 0.5250, 0.5025],
                    std=[0.2328, 0.2430, 0.2454],
                ),
                ToTensorV2(p=1.0),
            ])
            
def get_data():
    test_path = "./data_path/test_path.csv"
    test_set = Image_Dataset_(csv_path = test_path, train = False, transform=get_test_transforms())
    test_iter = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    return test_iter


def ensemble(models, model_num, test_iter, device):
    for i in range(model_num):
        models[i].eval()

    with torch.no_grad():
        test_loss = 0
        pbar = tqdm(enumerate(test_iter), total=len(test_iter), position=0, leave=True)
        submission_csv = pd.read_csv("./input/data/eval/info.csv")

        for step, img in pbar:
            img = img.to(device)
            for i in range(model_num):
                if i == 0:
                    y_pred = models[i](img)
                else:
                    y_pred += models[i](img)
            softmax_value = F.softmax(y_pred)

            if step == 0:
                softmax_tensor = torch.cat((softmax_tensor, softmax_value), dim = 1)
            else:
                softmax_tensor = torch.cat((softmax_tensor, softmax_value), dim = 0)
            submission_csv["ans"][step] = int(torch.argmax(y_pred, 1))

    submission_csv = submission_csv.astype({'ans': 'int'})
    submission_csv.to_csv("./efficientnet_b4_ensenble.csv", index=False)


def ensemble_softmax(softmax_list):
    submission_csv = pd.read_csv("./input/data/eval/info.csv")
    for index in range(12600):
        for i in range(len(softmax_list)):
            if i == 0:
                y_pred = softmax_list[i][index] + torch.tensor([0, 0.4, 0, 0, 0.4, 0, 0, 0.4, 0, 0, 0.4, 0, 0, 0.4, 0, 0, 0.4, 0]).to("cuda:0")
            else:
                y_pred += softmax_list[i][index]
        submission_csv["ans"][index] = int(torch.argmax(y_pred, 0))
    submission_csv = submission_csv.astype({'ans': 'int'})
    submission_csv.to_csv("./ensemble_softmax/ensemble_weight_0.4_total.csv", index=False)
            
        

if __name__ == "__main__":
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    test_iter = get_data()
    models_name = ["/opt/ml/new_models/age55_efficientnet_b4.pt", "/opt/ml/new_models/age56_efficientnet_b4.pt", "/opt/ml/new_models/age57_efficientnet_b4.pt", "/opt/ml/new_models/age58_efficientnet_b4.pt", "/opt/ml/new_models/age59_efficientnet_b4.pt", "/opt/ml/new_models/efficientnet_b4.pt"]
    models = []
    for model_name in models_name:
        print(model_name)
        models.append(torch.load(model_name))
    ensemble(models, 6, test_iter, "cuda:0")
    softmax_list = []

    # for i in range(55, 61):
    #     softmax_list.append(torch.load(f"/opt/ml/ensemble_softmax/age{i}_softmax_tensor_total.pt"))
    # ensemble_softmax(softmax_list)


