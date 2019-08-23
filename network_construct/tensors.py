import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch.utils.data as tdata

import numpy as np
from scipy.ndimage import zoom

class tensors(Dataset):
    """Loads ABIDE data as pre-saved tensors."""
    def __init__(self,transform=None):
        dim=3
        if dim==2:
            path='/vulcan/scratch/mtang/datasets/ABIDE/cpac/torch/2d/'
            # path='/vulcan/scratch/mtang/datasets/ABIDE/raw_scans/torch/'
        elif dim==3:
            path='/vulcan/scratch/mtang/datasets/ABIDE/cpac/torch/3d/'
            # path='/vulcan/scratch/mtang/datasets/ABIDE/raw_scans/torch/'
        
        # training set
        self.trainSet=[]
        for i in range(52):
            with open(path+'train/train'+str(i)+'.pkl', 'rb') as f:
                input_tensor=pickle.load(f)
                (self.trainSet).append(input_tensor)
        # test set
        self.testSet=[]
        for i in range(13):
            with open(path+'test/test'+str(i)+'.pkl', 'rb') as f:
                input_tensor=pickle.load(f)
                (self.testSet).append(input_tensor)
        
        self.allData=self.trainSet+self.testSet
        self.transform=transform

    def __len__(self):
        return len(self.allData)

    def __getitem__(self,subID):
        data=self.allData[subID]
        array=data[0].numpy()
        img=zoom(array, (1, 1, 3/61, 224/61, 224/73))
        img=torch.Tensor(img)
        sample=[img, data[1]]
        # img, label=sample
        if self.transform:
            img = self.transform(img)
            sample=[img, data[1]]
        
        return sample

