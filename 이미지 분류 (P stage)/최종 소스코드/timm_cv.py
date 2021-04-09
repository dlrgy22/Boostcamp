import argparse
import time
import warnings

import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from config import Config
from timm_model import Model

from data_set_cv import Image_Dataset_
from focalloss import *
from custom_loss import *

from albumentations import *
from albumentations.pytorch.transforms import ToTensorV2

def get_config():
    parser = argparse.ArgumentParser(description="use timm models")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--img_size", default=32, type=int)
    parser.add_argument("--model_name", default="tf_efficientnet_b4", type=str)
    parser.add_argument("--pretrained", default=True, type=bool)

    args = parser.parse_args()

    config = Config(
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        IMG_SIZE=args.img_size,
        MODEL_NAME=args.model_name,
        PRETRAINED=args.pretrained
    )
    return config

def get_train_transforms(config):
  return Compose([
                  CenterCrop(380, 380),                 
                  IAAPerspective(p=0.5),
                  HorizontalFlip(p=0.5),
                  HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                  RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                  CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
                  OneOf([
                    RandomGridShuffle(grid=(2, 2), p=0.5),
                    RandomGridShuffle(grid=(4, 2), p=0.5)
                  ], p=0.5),
                  Normalize(
                    mean=[0.5610, 0.5250, 0.5025],
                    std=[0.2328, 0.2430, 0.2454],
                ),
                  ToTensorV2(p=1.0),
  ])

def get_test_transforms(config):
    return Compose([
                CenterCrop(380, 380),
                Normalize(
                    mean=[0.5610, 0.5250, 0.5025],
                    std=[0.2328, 0.2430, 0.2454],
                ),
                ToTensorV2(p=1.0)
            ])
            
            
def get_data(config, train_df, val_df, test_df):
                                                            
    train_set = Image_Dataset_(df = train_df, transform=get_train_transforms(config))
    val_set = Image_Dataset_(df = val_df, transform=get_test_transforms(config))
    test_set = Image_Dataset_(df = test_df, train = False, transform=get_test_transforms(config))

    train_iter = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)
    val_iter = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)
    test_iter = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    return train_iter, val_iter, test_iter


def get_model(config):
    network = Model(model_name=config.MODEL_NAME, n_classes=18, pretrained=config.PRETRAINED, dropRate=0.5)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    criterion = CustomLoss().to(config.device)

    return network, optimizer, scaler, scheduler, criterion


def train_one_epoch(epoch, model, loss_fn, optim, scaler, train_loader, device, beta=1.0, cutmix_prop=0.5, scheduler=None):
    model.train()

    t = time.time()
    running_loss = 0
    sample_num = 0
    preds_all = []
    targets_all = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device).float()
        labels = labels.to(device).long()

        with autocast():
            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prop:
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(imgs.size()[0]).cuda()
                label_a = labels
                label_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                preds = model(imgs.float())

                loss = loss_fn(preds, label_a) * lam + loss_fn(preds, label_b) * (1. - lam)

            else:    
                preds = model(imgs.float())
                loss = loss_fn(preds, labels)

      
            preds_all += [torch.argmax(preds, 1).detach().cpu().numpy()]
            targets_all += [labels.detach().cpu().numpy()]
            scaler.scale(loss).backward()

            running_loss += loss.item()*labels.shape[0]
            sample_num += labels.shape[0]

            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            description = f"epoch {epoch} loss: {running_loss/sample_num: .4f}"
            pbar.set_description(description)
          
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    accuracy = (preds_all == targets_all).mean()
    print("     train accuracy = {:.4f}".format(accuracy))

    if scheduler is not None:
        scheduler.step()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def test_one_epoch(epoch, model, loss_fn, test_loader, device):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    preds_all = []
    targets_all = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device).float()
        labels = labels.to(device).long()
    
        preds = model(imgs.float())
        preds_all += [torch.argmax(preds, 1).detach().cpu().numpy()]
        targets_all += [labels.detach().cpu().numpy()]

        loss = loss_fn(preds, labels)

        loss_sum += loss.item()*labels.shape[0]
        sample_num += labels.shape[0]
    
        description = f"epoch {epoch} loss: {loss_sum/sample_num:.4f}"
        pbar.set_description(description)

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    accuracy = (preds_all == targets_all).mean()
    print("     test accuracy = {:.4f}".format(accuracy))

    return accuracy


def train_model(model, loss_fn, optimizer, scaler, train_loader, test_loader, scheduler, config, save_path):
    prev_acc = 0
    for epoch in range(config.EPOCHS):
        epoch_acc = train_one_epoch(epoch, model, loss_fn, optimizer, scaler, train_loader, config.device, scheduler=scheduler)
        with torch.no_grad():
            test_acc = test_one_epoch(epoch, model, loss_fn, test_loader, config.device)
            if test_acc > prev_acc:
                torch.save(model, save_path)
                prev_acc = test_acc


def make_submission(test_loader, model, submission_path):
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
    submission_csv = pd.read_csv("./input/data/eval/info.csv")
    with torch.no_grad():
        model.eval()
        for step, img in pbar:
            y_pred = model(img.cuda())
            submission_csv["ans"][step] = int(torch.argmax(y_pred, 1))
    submission_csv = submission_csv.astype({'ans': 'int'})
    submission_csv.to_csv(submission_path, index=False)


def get_cv_df(index, df):
    cv_val = df[df["groups"] == index]
    index_name = cv_val.index
    cv_train = df.drop(index_name)

    return cv_val, cv_train


if __name__ == "__main__":
    config = get_config()
    for age in range(55, 57):
        df = pd.read_csv(f"./data_path/new_mytrain_age{age}.csv")
        test_df = pd.read_csv("./data_path/test_path.csv")
        train_df = []
        val_df = []
        for i in range(5):
            cv_val, cv_train = get_cv_df(i, df)
            train_df.append(cv_train)
            val_df.append(cv_val)
        
        for i in range(5):
            print(f"cv {i}")
            #submission_path = f"./submission/cv{i}_efficientnet_b4_age.csv"
            save_path = f'./new_models/age{age}_cv{i}_efficientnet_b4_high.pt'

            train_iter, val_iter, test_iter = get_data(config, train_df[i], val_df[i], test_df)
            model, optimizer, scaler, scheduler, loss_function = get_model(config)
            model.cuda()
            train_model(model ,loss_function, optimizer, scaler, train_iter, val_iter, scheduler, config, save_path)
            torch.save(model, f'./new_models/age{age}_cv{i}_efficientnet_b4.pt')
