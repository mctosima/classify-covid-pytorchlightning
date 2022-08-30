'''import'''
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

'''augmentation must same between both eyes'''
aug = A.Compose([
    A.Resize(256,256),
    A.CenterCrop(224,222),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.ShiftScaleRotate(0.05,0.05,5),
    A.RandomBrightnessContrast(),
    A.CLAHE(),
    A.CoarseDropout(),
    A.Normalize(),
    ToTensorV2(),
],p=1.0, additional_targets={'image0':'image'})

'''Define the Path'''
train_df = pd.read_csv('data/ocular/full_df.csv')
train_path = 'data/ocular/ODIR-5K/ODIR-5K/Training_Images/'
test_path = 'data/ocular/ODIR-5K/ODIR-5K/Testing_Images/'

'''DataReader'''
class DataReader(Dataset):
    def __init__(self,df,path,transform=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        # read id from dataframe and find the image path
        sub_id = self.df.iloc[index,0]
        left_path = os.path.join(self.path+str(sub_id)+'_left.jpg')
        right_path = os.path.join(self.path+str(sub_id)+'_right.jpg')

        # read image and convert to RGB
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # read labels from dataframe
        labels = self.df.iloc[index,7:15].astype(int)

        # apply transformation
        if self.transform:
            image = self.transform(image=left_img,image0=right_img)
            left_eye = image['image']
            right_eye = image['image0']
        return left_eye, right_eye, labels.to_numpy()

train_data = DataReader(train_df,train_path,transform=aug)
train_loader = DataLoader(train_data,shuffle=False,batch_size=16,num_workers=0)
left,right,label = next(iter(train_loader))


'''Try to show the image'''
# plt.figure(figsize=(16,16))
# grid_img = torchvision.utils.make_grid(left,8,4)
# plt.imshow(grid_img.permute(1,2,0))
# plt.savefig('junk.png')

# plt.figure(figsize=(16,16))
# grid_img = torchvision.utils.make_grid(right,8,4)
# plt.imshow(grid_img.permute(1,2,0))
# plt.savefig('junk2.png')


'''
about the dataset:
https://odir2019.grand-challenge.org
'''

