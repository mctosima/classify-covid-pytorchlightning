""" import modules """
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
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import seed_everything, LightningModule, Trainer
from sklearn.metrics import classification_report
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

wandb_logger = WandbLogger(project="covid-scan-classification", log_model=False)


class OurModel(LightningModule):
    def __init__(self):
        super().__init__()

        # model architecture
        self.model = models.resnet18(weights="DEFAULT")
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # param
        self.lr = 0.001
        self.batch_size = 224
        self.num_workers = 12

        # loss function
        self.criterion = nn.MSELoss()

        # evaluation metrics
        self.acc = torchmetrics.Accuracy()

        # loss curve and accuracy curve
        self.trainacc, self.valacc = [], []
        self.trainloss, self.valloss = [], []

        # Augmentation
        self.aug = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # load data
        self.dataset = torchvision.datasets.ImageFolder(
            "data/MRI_Covid", transform=self.aug
        )
        self.train_size = int(np.floor(0.85 * len(self.dataset)))
        self.val_size = int(np.floor(0.05 * len(self.dataset)))
        self.test_size = len(self.dataset) - self.train_size - self.val_size
        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, [self.train_size, self.val_size, self.test_size]
        )

        # wandblogger
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self):
        dl = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dl

    def training_step(self, batch, batch_idx):
        image, label = batch
        pred = self(image)
        loss = self.criterion(pred.flatten(), label.float())
        acc = self.acc(pred.flatten(), label)
        # wandblogger
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/acc", acc, on_epoch=True, on_step=False)
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

        print(f"Training loss: {loss} | Training accuracy: {acc}")

    def val_dataloader(self):
        dl = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return dl

    def validation_step(self, batch, batch_idx):
        image, label = batch
        pred = self(image)
        loss = self.criterion(pred.flatten(), label.float())
        acc = self.acc(pred.flatten(), label)
        # early stopping and wandblogger
        self.log("val/acc", acc, on_epoch=True, on_step=False)
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, outputs):
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
        self.valloss.append(loss)
        self.valacc.append(acc)

        print(f"VALIDATING --> Validation loss: {loss} | Validation accuracy: {acc}")

    def test_dataloader(self):
        dl = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return dl

    def test_step(self, batch, batch_idx):
        image, label = batch
        pred = self(image)
        loss = self.criterion(pred.flatten(), label.float())
        acc = self.acc(pred.flatten(), label)
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


model = OurModel()

# wandblogger
wandb_logger.watch(model)  # to watch the model

# checkpoint_callback = ModelCheckpoint(monitor='val/acc', mode='max')

trainer = Trainer(
    devices=1,
    accelerator="gpu",
    max_epochs=20,
    deterministic=True,
    num_sanity_val_steps=0,
    callbacks=[
        EarlyStopping(monitor="train/loss", patience=2, min_delta=0.01),
        #  checkpoint_callback
    ],
    logger=wandb_logger,
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    #   limit_train_batches=20,
    #   limit_val_batches=5,
)
trainer.fit(model)

trainer.validate(model)

# test step
trainer.test(model)
