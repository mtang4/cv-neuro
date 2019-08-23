from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import nibabel as nib
import os
import pandas as pd
from pandas import ExcelWriter, ExcelFile

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

class fmri(Dataset):
    """Loads func_preproc derivatives of CPAC preprocessed data + corresponding ROI time series."""
    def __init__(self, transform=None):
        self.path='/vulcan/scratch/mtang/datasets/ABIDE/fmri/'
        self.roi='/vulcan/scratch/mtang/datasets/ABIDE/ROI/'
        df=pd.read_excel('/vulcan/scratch/mtang/code/neuroimaging/preprocessing/abide_summary.xlsx', sheet_name='summary')
        self.subjects=df['FILE_ID']
        self.labels=df['DX_GROUP']
        self.transform=transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,subID):
        name=self.subjects.iloc[subID]
        img_name=self.path+self.subjects.iloc[subID]+'.nii.gz'
        img=nib.load(img_name)

        # save img data array
        imgData=img.get_fdata()
        newImg=torch.Tensor(imgData)
        newImg=newImg.permute(3,0,1,2)
        
        # save ROI time series
        allvoxels=[]
        linenum=0
        with open(self.roi+self.subjects.iloc[subID]+'.1D') as f:
            # read data
            for line in f:
                data=line.split('\t')
                if linenum>0:
                    data=[float(i) for i in data]
                    allvoxels.append(data)
                linenum+=1
                        
            # create ROI time series
            allvoxels=np.array(allvoxels)
            allvoxels=allvoxels.transpose()

        # get output label
        img_label=self.labels.iloc[subID]-1

        # construct output as [fMRI data, ROI data, label]
        allvoxels=np.transpose(allvoxels)
        sample = [newImg, allvoxels, img_label]

        if self.transform:
            sample = self.transform(sample)
        
        return sample