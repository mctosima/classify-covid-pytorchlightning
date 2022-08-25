''' import modules '''
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


class OurModel(LightningModule):
    def __init__(self):
        super().__init__()
        
        # model architecture
        self.model = models.resnet18()
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 224),
            nn.ReLU(),
            nn.Linear(224,1),
        )
    
        # param
        self.lr = 0.001
        self.batch_size = 224
        self.num_workers = 4
    
        # loss function
        self.criterion = nn.BCEWithLogitsLoss()
    
        # evaluation metrics
        self.acc = torchmetrics.Accuracy()
    
        # loss curve and accuracy curve
        self.trainacc, self.valacc = [], []
        self.trainloss, self.valloss = [], []

        # Augmentation
        self.aug = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
        # load data
        self.dataset = torchvision.datasets.ImageFolder('data', transform=self.aug)
        self.train_size = int(np.floor(0.8 * len(self.dataset)))
        self.val_size = len(self.dataset) - self.train_size
        self.train_set, self.val_set = random_split(self.dataset, [self.train_size, self.val_size])
        # split data
    
    def forward(self,x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt
    
    def train_dataloader(self):
        dl = DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return dl
    
    def training_step(self,batch,batch_idx): 
        image, label = batch
        pred = self(image)
        loss = self.criterion(pred.flatten(), label.float())
        acc = self.acc(pred.flatten(), label)
        return {'loss': loss, 'acc': acc}
    
    def training_epoch_end(self,outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean().detach().cpu().numpy()
        acc = torch.stack([x['acc'] for x in outputs]).mean().detach().cpu().numpy()
        self.trainloss.append(loss)
        self.trainacc.append(acc)
    
    def val_dataloader(self):
        dl = DataLoader(self.val_set, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return dl
    
    def val_step(self,batch,batch_idx):
        image, label = batch
        pred = self(image)
        loss = self.criterion(pred.flatten(), label.float())
        acc = self.acc(pred.flatten(), label)
        return {'loss': loss, 'acc': acc}
    
    def val_epoch_end(self,outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean().detach().cpu().numpy()
        acc = torch.stack([x['acc'] for x in outputs]).mean().detach().cpu().numpy()
        self.trainloss.append(loss)
        self.trainacc.append(acc)
        print(f'Validation loss: {loss} | Validation accuracy: {acc}')
    
model = OurModel()
seed_everything(0)
trainer = Trainer(gpus=1,
                  max_epochs=1,
                  progress_bar_refresh_rate=0,
                  deterministic=True,
                  accelerator='mps',
                  precision=16,
                  num_sanity_val_steps=2,
                  limit_train_batches=20,
                  limit_val_batches=5,
                  )
trainer.fit(model)
        
    