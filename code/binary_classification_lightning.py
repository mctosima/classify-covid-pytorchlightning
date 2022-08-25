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

'''augmentation'''
aug = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

'''load dataset'''
dataset = torchvision.datasets.ImageFolder('data', transform=aug)

'''split dataset into train and test'''
train_size = int(np.floor(0.8 * len(dataset)))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
print(f'Train set size: {len(train_set)} | Val set size: {len(val_set)}')

'''create dataloaders'''
batch_size = 32
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size, shuffle=True, num_workers=4)

''' show one batch of train_loader '''
# plotonebatch = next(iter(train_loader))
# plt.figure()
# grid_img = torchvision.utils.make_grid(plotonebatch[0],8,4)
# plt.imshow(grid_img.permute(1,2,0))
# plt.show()

''' Load Model '''
model = models.resnet18()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 224),
    nn.ReLU(),
    nn.Linear(224,1),
)

''' get predictions for single pass'''
pred = model(next(iter(train_loader))[0])

criterion = nn.BCEWithLogitsLoss()
loss1p = criterion(pred.flatten(), next(iter(train_loader))[1].float())
print(f'Loss for single pass: {loss1p}')

''' get accuracy for single pass'''
acc = torchmetrics.Accuracy()
acc1p = acc(pred.flatten(), next(iter(train_loader))[1])
print(f'Accuracy for single pass: {acc1p}')

''' ------------------------------ '''
''' Using PyTorch Lightning '''
class OurClass(LightningModule):
    def __init__(self):
        super().__init__()
        
        # model architecture
        self.model = models.resnet18()
        self.model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 224),
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
        self.trainloss, self.valloss = [], [0]

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
        
        pass
    
    def training_epoch_end(self,outputs):
        pass
    
    def val_dataloader(self):
        dl = DataLoader(self.val_set, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return dl
    
    def val_step(self,batch,batch_idx):
        pass
    
    def val_epoch_end(self,outputs):
        pass
    
        
    