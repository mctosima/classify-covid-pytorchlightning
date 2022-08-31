"""import"""
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, Dataset

import torchmetrics
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
import torchvision

import pandas as pd
import numpy as np
import os
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

"""
about the dataset:
https://odir2019.grand-challenge.org
"""

"""augmentation must same between both eyes"""
aug = A.Compose(
    [
        A.Resize(256, 256),
        A.CenterCrop(224, 222),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ShiftScaleRotate(0.05, 0.05, 5),
        A.RandomBrightnessContrast(),
        A.CLAHE(),
        A.CoarseDropout(),
        A.Normalize(),
        ToTensorV2(),
    ],
    p=1.0,
    additional_targets={"image0": "image"},
)

"""Define the Path"""
train_df = pd.read_csv("data/ocular/full_df.csv")
train_path = "data/ocular/ODIR-5K/ODIR-5K/Training_Images/"
test_path = "data/ocular/ODIR-5K/ODIR-5K/Testing_Images/"

"""DataReader"""


class DataReader(Dataset):
    def __init__(self, df, path, transform=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # read id from dataframe and find the image path
        sub_id = self.df.iloc[index, 0]
        left_path = os.path.join(self.path + str(sub_id) + "_left.jpg")
        right_path = os.path.join(self.path + str(sub_id) + "_right.jpg")

        # read image and convert to RGB
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # read labels from dataframe
        labels = self.df.iloc[index, 7:15].astype(int)

        # apply transformation
        if self.transform:
            image = self.transform(image=left_img, image0=right_img)
            left_eye = image["image"]
            right_eye = image["image0"]
        return left_eye, right_eye, labels.to_numpy()


train_data = DataReader(train_df, train_path, transform=aug)
train_loader = DataLoader(train_data, shuffle=False, batch_size=16, num_workers=0)
left, right, label = next(iter(train_loader))


"""Try to show the image"""
# plt.figure(figsize=(16,16))
# grid_img = torchvision.utils.make_grid(left,8,4)
# plt.imshow(grid_img.permute(1,2,0))
# plt.savefig('junk.png')

# plt.figure(figsize=(16,16))
# grid_img = torchvision.utils.make_grid(right,8,4)
# plt.imshow(grid_img.permute(1,2,0))
# plt.savefig('junk2.png')

"""Split the data into train and val"""
from sklearn.model_selection import train_test_split
from sklearn import metrics

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=21)


"""
Define the model
The model: https://github.com/rwightman/pytorch-image-models by Ross Wightman
"""

import timm


class Resnet18Timm(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True)
        self.fc1 = nn.Linear(2000, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 8)

        # param
        self.lr = 0.001
        self.batch_size = 96
        self.num_workers = 8
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, left_image, right_image):
        """
        Explanation:
            - each eye image passed to the model and the output is concatenated
            - the output is passed to the fully connected layer with 2000 neurons input (since the model has 2 eyes)
            - the last layer has 1000 neurons and the output is passed to the fully connected layer with 8 neurons
        """

        left = self.model(left_image)
        right = self.model(right_image)
        x = torch.cat((left, right), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def focalloss(self, BCE, alpha=0.75, gamma=2):
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
        return focal_loss.mean()

    def configure_optimizers(self):
        opt = torch.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(
            opt, T_0=5, T_mult=1, eta_min=1e-5, last_epoch=-1
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def train_dataloader(self):
        dl = DataLoader(
            DataReader(train_df, train_path, transform=aug),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return dl

    def training_step(self, batch, batch_idx):
        left, right, label = batch
        pred = self(left, right)
        loss = self.focalloss(self.criterion(pred, label))
        self.log("train_loss", loss, on_epoch=True, on_step=False)

    def training_epoch_end(self, outputs):
        pass

    def val_dataloader(self):
        dl = DataLoader(
            DataReader(val_df, train_path, transform=aug),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return dl

    def val_step(self, batch, batch_idx):
        left, right, label = batch
        pred = self(left, right)
        loss = self.focalloss(self.criterion(pred, label))
        self.log("val_loss", loss, on_epoch=True, on_step=False)

    def val_epoch_end(self, outputs):
        pass

    def test_dataloader(self):
        dl = DataLoader(
            DataReader(val_df, train_path, transform=aug),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return dl

    def test_step(self, batch, batch_idx):
        left, right, label = batch
        out = self(left, right)
        return {
            "label": label.detach().cpu().numpy(),
            "out": out.detach().cpu().numpy(),
        }

    def ODIR_Metrics(self, gt_data, pr_data):
        th = 0.5
        gt = gt_data.flatten()
        pr = pr_data.flatten()
        kappa = metrics.cohen_kappa_score(gt, pr > th)
        f1 = metrics.f1_score(gt, pr > th, average="micro")
        auc = metrics.roc_auc_score(gt, pr)
        final_score = (kappa + f1 + auc) / 3
        return kappa, f1, auc, final_score

    def test_epoch_end(self, outputs):
        label = torch.cat([x["label"] for x in outputs])
        pred = torch.cat([x["out"] for x in outputs])
        kappa, f1, auc, final_score = self.ODIR_Metrics(label, pred)


model = Resnet18Timm()
