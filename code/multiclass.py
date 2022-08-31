""" import modules """
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torch.nn as nn
import torchmetrics
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
    ConcatDataset,
    SubsetRandomSampler,
)
from pytorch_lightning import seed_everything, LightningModule, Trainer
from sklearn.metrics import classification_report
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import cv2

"""import albumentations"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""augmentations arguments"""
img_size = 512

"""wandb logger"""
wandb_logger = WandbLogger(project="covid-xray", log_model=False)

"""compose augmentations"""
aug = A.Compose(
    [
        A.Resize(img_size + 32, img_size + 32),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ShiftScaleRotate(),
        A.Blur(),
        A.RandomGamma(),
        A.Sharpen(),
        A.GaussNoise(),
        A.CoarseDropout(8, 64, 64),
        A.CLAHE(),
        A.Normalize(mean=0, std=1),
        ToTensorV2(),
    ]
)


class DataReader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x = self.dataset[index][0]
        y = self.dataset[index][1]

        if self.transform:
            x = np.array(x)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            x = self.transform(image=x)["image"]

        return x, y

    def __len__(self):
        return len(self.dataset)


"""
get the model from: https://github.com/mlmed/torchxrayvision
Get the dataset from: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia
"""

import torchxrayvision as xrv

"""The Class"""


class XrayVision(LightningModule):
    def __init__(self, combined, train_set, val_set, test_set):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.combined = combined

        # model architecture
        self.model = xrv.models.ResNet(weights="resnet50-res512-all")
        self.model.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=3),
        )
        self.model = self.model.model

        # param
        self.lr = 0.001
        self.batch_size = 16
        self.num_workers = 4

        # var for loss and acc
        self.trainacc, self.valacc = [], []
        self.trainloss, self.valloss = [], []

        # loss fn
        self.criterion = nn.CrossEntropyLoss()

        # metrics
        self.acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val/loss"}

    def train_dataloader(self):
        dl = DataLoader(
            DataReader(self.combined, aug),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.train_set,
        )
        return dl

    def training_step(self, batch, batch_idx):
        image, label = batch
        pred = self(image)
        loss = self.criterion(pred, label)
        acc = self.acc(pred, label)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "acc": acc}

    def training_epoch_end(self, outputs):
        loss = (
            torch.stack([x["loss"] for x in outputs])
            .mean()
            .detach()
            .cpu()
            .numpy()
            .round(2)
        )
        acc = (
            torch.stack([x["acc"] for x in outputs])
            .mean()
            .detach()
            .cpu()
            .numpy()
            .round(2)
        )
        self.trainloss.append(loss)
        self.trainacc.append(acc)

        # wandblogger
        print(f"Training loss: {loss} | Training accuracy: {acc}")

    def val_dataloader(self):
        dl = DataLoader(
            DataReader(self.combined, aug),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.val_set,
        )
        return dl

    def validation_step(self, batch, batch_idx):
        image, label = batch
        pred = self(image)
        loss = self.criterion(pred, label)
        acc = self.acc(pred, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True)
        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, outputs):
        loss = (
            torch.stack([x["loss"] for x in outputs])
            .mean()
            .detach()
            .cpu()
            .numpy()
            .round(3)
        )
        acc = (
            torch.stack([x["acc"] for x in outputs])
            .mean()
            .detach()
            .cpu()
            .numpy()
            .round(3)
        )
        self.valloss.append(loss)
        self.valacc.append(acc)
        print(f"VALIDATING --> Validation loss: {loss} | Validation accuracy: {acc}")

    def test_dataloader(self):
        dl = DataLoader(
            DataReader(test_set),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return dl

    def test_step(self, batch, batch_idx):
        image, label = batch
        pred = self(image)
        loss = self.criterion(pred, label)
        acc = self.acc(pred, label)
        return {"loss": loss, "acc": acc}

    def test_epoch_end(self, outputs):
        loss = (
            torch.stack([x["loss"] for x in outputs])
            .mean()
            .detach()
            .cpu()
            .numpy()
            .round(2)
        )
        acc = (
            torch.stack([x["acc"] for x in outputs])
            .mean()
            .detach()
            .cpu()
            .numpy()
            .round(2)
        )

        wandb.log({"test/loss": loss, "test/acc": acc})
        print(f" Testing loss: {loss} | Testing accuracy: {acc}")


"""Dataset Preparation"""
train_dataset = torchvision.datasets.ImageFolder("data/Covid19_XRAY/train")
train_size = int(np.floor(0.9 * len(train_dataset)))
val_size = len(train_dataset) - train_size
train_set, val_set = random_split(train_dataset, [train_size, val_size])
test_set = torchvision.datasets.ImageFolder("data/Covid19_XRAY/test")

"""Creating the K-Fold"""
from sklearn.model_selection import KFold

combined_set = ConcatDataset([train_set, val_set])
kfold = KFold(n_splits=5, shuffle=True, random_state=21)


"""Trainer"""
# model = XrayVision(train_set,val_set,test_set)
# trainer = Trainer(max_epochs=1,
#                   limit_train_batches=5,
#                   limit_val_batches=5,
#                   devices=1,
#                   accelerator='gpu',
#                   )

# trainer.fit(model)

for fold, (train_idx, val_idx) in enumerate(kfold.split(combined_set)):
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)

    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename=f"fold_{fold}_checkpoint.pth",
    )
    earlystopping = EarlyStopping(monitor="val/loss", mode="min", patience=5)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model = XrayVision(combined_set, train_subsampler, val_subsampler, test_set)

    wandb_logger.watch(model)  # to watch the model in wandb

    trainer = Trainer(
        max_epochs=15,
        deterministic=True,
        gpus=1,
        precision=16,
        accumulate_grad_batches=4,
        callbacks=[lr_monitor, earlystopping, checkpoint],
        num_sanity_val_steps=0,
        limit_train_batches=5,
        limit_val_batches=1,
        logger=wandb_logger,
        log_every_n_steps=1,
    )

    trainer.fit(model)
    torch.save(model.state_dict(), "checkpoints/last_{}.pth".format(fold))
