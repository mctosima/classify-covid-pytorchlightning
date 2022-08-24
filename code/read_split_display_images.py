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

# augmentation
aug = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load dataset
dataset = torchvision.datasets.ImageFolder('data', transform=aug)

# split dataset into train and test
train_size = int(np.floor(0.8 * len(dataset)))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
print(f'Train set size: {len(train_set)} | Val set size: {len(val_set)}')

# create dataloaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=4)

# show one batch of train_loader
plotonebatch = next(iter(train_loader))
plt.figure()
grid_img = torchvision.utils.make_grid(plotonebatch[0],8,4)
plt.imshow(grid_img.permute(1,2,0))
plt.show()