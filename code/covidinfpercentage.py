"""Import"""
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import WandbLogger

import torchmetrics
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
import torchvision
from torchinfo import summary

import timm
import pandas as pd
import numpy as np
import os
import cv2
from glob import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

"""CSV and Prepare Datapath"""
path = "data/covid19_infection_percentage_est/"
# train_path = sorted(glob(os.path.join(path,"Train","*.png")))
train_df = pd.read_csv("data/covid19_infection_percentage_est/Train.csv", header=None)

"""DataReader Class"""


class DataReader(Dataset):
    def __init__(self, df, path, transform=None, folder="Train"):
        super().__init__()
        self.df = df
        self.path = path
        self.transform = transform
        self.folder = folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index, 0]
        img_path = os.path.join(self.path, self.folder, filename)

        # read image and augmentation
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]

        # read label
        label = self.df.iloc[index, 1]

        return img, label.flatten().astype(np.float32)


"""Augmentation"""
aug = A.Compose(
    [
        A.Resize(224 + 32, 224 + 32),
        A.CenterCrop(224, 224),
        A.HorizontalFlip(0.5),
        A.ShiftScaleRotate(0.05, 0.05, 5),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)


"""Split Train and Validation"""
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=24)

"""Check One Batch Data"""
train_data = DataReader(train_df, path, transform=aug)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
img, label = next(iter(train_loader))

# plt.figure(figsize=(8, 3))
# grid_img = torchvision.utils.make_grid(img, nrow=4)
# plt.imshow(grid_img.permute(1, 2, 0))
# plt.show()


"""Create Class Model"""


class Resnet18TimModified(LightningModule):
    def __init__(self, train_df, val_df):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df

        # model architecture
        self.model = timm.create_model("resnest50d", pretrained=True)
        self.fc = nn.Sequential(
            nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 250), nn.Linear(250, 1)
        )

        # param
        self.lr = 0.001
        self.batch_size = 32
        self.num_workers = 8

        self.acc = torchmetrics.Accuracy()

        # loss function
        self.criterion = nn.L1Loss()

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
        return [opt], [scheduler]

    def train_dataloader(self):
        dl = DataLoader(
            DataReader(self.train_df, path, transform=aug),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return dl

    def training_step(self, batch, batch_idx):
        image, label = batch
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        # acc = self.acc(pred, label)
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        # self.log("train/acc", acc, on_step=True, on_epoch=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        pass

    def val_dataloader(self):
        dl = DataLoader(
            DataReader(self.val_df, path, transform=aug),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return dl

    def validation_step(self, batch, batch_idx):
        image, label = batch
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        acc = self.acc(pred.int(), label.int())
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "acc": acc}


"""Train Fit"""
model = Resnet18TimModified(train_df, val_df)
wandb_logger = WandbLogger(project="covid-inf_percentage", log_model=False)
wandb_logger.watch(model)

progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
    )
)

trainer = Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    precision=16,
    deterministic=True,
    logger=wandb_logger,
    log_every_n_steps=15,
    fast_dev_run=False,
    callbacks=progress_bar,
)

trainer.fit(model)

"""Test"""

"""write the submission csv"""
