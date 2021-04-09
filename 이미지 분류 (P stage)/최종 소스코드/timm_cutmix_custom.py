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

from adamp import AdamP
from data_set import Image_Dataset_
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
                  HorizontalFlip(p=0.5),
                  HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                  RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
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


def get_data(config, train_path, test_path):
                                                            
    train_set = Image_Dataset_(csv_path = train_path, transform=get_train_transforms(config))
    val_set = Image_Dataset_(csv_path = val_path, transform=get_test_transforms(config))
    test_set = Image_Dataset_(csv_path = test_path, train = False, transform=get_test_transforms(config))

    train_iter = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)
    val_iter = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)
    test_iter = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    return train_iter, test_iter


def get_model(config):
    network = Model(model_name=config.MODEL_NAME, n_classes=18, pretrained=config.PRETRAINED, dropRate=0.5)
    optimizer = AdamP(network.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    criterion = CustomLoss().to(config.device)

    return network, optimizer, scaler, scheduler, criterion


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

def train_one_epoch(epoch, model, loss_fn, optim, scaler, train_loader, old_train_loader, device, beta=1.0, cutmix_prop=0.5, scheduler=None):
    model.train()

    t = time.time()
    running_loss = 0
    sample_num = 0
    preds_all = []
    targets_all = []

    old_iter = iter(old_train_loader)
    old_length = len(old_train_loader)
    old_step = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device).float()
        labels = labels.to(device).long()

        with autocast():
            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prop:
                if old_step < old_length-1:
                    old_imgs, old_labels = next(old_iter)
                    old_imgs = old_imgs.to(device).float()
                    old_labels = old_labels.to(device).long()
                    old_step += 1

                else:
                    old_step = 1
                    old_iter = iter(old_train_loader)
                    old_imgs, old_labels = next(old_iter)
                    old_imgs = old_imgs.to(device).float()
                    old_labels = old_labels.to(device).long()
                

                imgs, custom_labels = custom_cutmix(imgs, labels, old_imgs, old_labels, 1.)
                preds = model(imgs.float())
                loss = loss_fn(preds, custom_labels[0]) * custom_labels[2] + loss_fn(preds, custom_labels[1]) * (1. - custom_labels[2])

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


def custom_cutmix(data, target, old_data, old_target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_old_target = old_target[indices]
    
    size = data.size()
    W = size[2]
    new_data = data.clone()
    new_data[:, :, :, :W//2] = old_data[indices, :, :, W//2:]
    # adjust lambda to exactly match pixel ratio
    lam = 0.5
    targets = (target, shuffled_old_target, lam)

    return new_data, targets


def train_model(model, loss_fn, optimizer, scaler, train_loader, old_train_loader, scheduler, config, save_path):
    prev_acc = 0
    for epoch in range(config.EPOCHS):
        epoch_acc = train_one_epoch(epoch, model, loss_fn, optimizer, scaler, train_loader, old_train_loader, config.device, scheduler=scheduler)
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


if __name__ == "__main__":
    config = get_config()
    for i in range(55, 60):
        train_path = f"./data_path/age{i}_train_path.csv"
        test_path = "./data_path/test_path.csv"
        val_path = f"./data_path/age{i}_val_path.csv"
        old_path = f"/opt/ml/data_path/age{i}_old_path.csv"

        submission_path = f"./submission/efficientnet_b4_age{i}.csv"
        save_path = f'./new_models/age{i}_efficientnet_b4_total_high.pt'

        old_set = Image_Dataset_(csv_path = old_path, transform=get_train_transforms(config))
        old_iter = DataLoader(old_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)

        train_iter, test_iter = get_data(config, train_path, test_path)
        model, optimizer, scaler, scheduler, loss_function = get_model(config)
        model.cuda()
        train_model(model ,loss_function, optimizer, scaler, train_iter, old_iter, scheduler, config, save_path)

        torch.save(model, f'./new_models/age{i}_efficientnet_b4_total.pt')
        # model = torch.load('/opt/ml/models/age58_custom_loss_high.pt')
        make_submission(test_iter, model, submission_path)
