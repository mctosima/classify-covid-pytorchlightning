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

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import cv2

from glob import glob
from sklearn.model_selection import train_test_split
import nibabel as nib

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

"""models and loss functions source
https://smp.readthedocs.io/en/latest/
"""


"""
Dataset Source
https://data.mendeley.com/datasets/zm6bxzhmfz
"""

data_path = "data/hipjoints/"
images = sorted(glob(data_path + "images/*.nii.gz"))
labels = sorted(glob(data_path + "labels/*.nii.gz"))
data_dicts = [
    {"image": img_name, "label": label_name}
    for img_name, label_name in zip(images, labels)
]

""" Split Data """
train_files, val_files = train_test_split(data_dicts, test_size=0.2, random_state=0)

""" Read One Images """
# oneimg = nib.load(
#     train_files[0]["image"]
# ).get_fdata()  # get_fdata() returns a numpy array
# onelbl = nib.load(train_files[0]["label"]).get_fdata()
# print("==DATASET INFORMATION==")
# print(f"Sample label: {np.unique(onelbl)}")
# print(f"Image Shape: {oneimg.shape} | File type: {type(oneimg)}")
# print(f"Label Shape: {onelbl.shape} | File type: {type(onelbl)}")
# print(f"Max Label: {onelbl.max()} | Min Label: {onelbl.min()}")

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(np.squeeze(oneimg, -1), cmap="gray")
# ax[1].imshow(np.squeeze(onelbl, -1), cmap="jet")
# plt.show()

"""Augmentation"""
train_aug = A.Compose(
    [
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(10),
        A.GaussNoise(p=0.1),
        A.Normalize(mean=(0), std=(1)),
        ToTensorV2(p=1.0),
    ]
)

val_aug = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(mean=(0), std=(1)),
        ToTensorV2(p=1.0),
    ]
)

"""Datareader"""


class DataReader(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["image"]
        label_path = self.data[idx]["label"]
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        transformed = self.transform(image=image, label=label)
        image = transformed["image"]
        label = transformed["label"]
        label = torch.moveaxis(label, 2, 0)

        return image, label


class SegmentationModel(LightningModule):
    def __init__(self):
        super().__init__()

        # model architec
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=3,
        )

        # param
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 16

        # loss function
        self.loss = DiceLoss(mode="multiclass", classes=3, from_logits=True)

        # evaluation metrics
        self.iou = torchmetrics.IoU(num_classes=3)

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-5, last_epoch=-1
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def train_dataloader(self):
        data = DataReader(train_files, train_aug)
        train_loader = DataLoader(
            data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        image, segment = batch[0], torch.squeeze(batch[1], 1).long()
        out = self.model(image)
        loss = self.loss(out, segment)
        iou = self.iou(out, segment)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "iou": iou}

    def training_epoch_end(self, outputs):
        pass

    def val_dataloader(self):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass


"""Trainer"""
lr_monitor = LearningRateMonitor(logging_interval="epoch")
model = SegmentationModel()
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=1,
    callbacks=[lr_monitor],
)

trainer.fit(model)
