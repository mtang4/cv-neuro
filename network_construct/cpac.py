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


# cpac preprocessed data
class cpac(Dataset):
	"""ABIDE neuroimaging dataset."""
	def __init__(self,transform=None):
		"""
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
		df=pd.read_excel('/vulcan/scratch/mtang/code/neuroimaging/preprocessing/abide_summary.xlsx', sheet_name='summary')
		self.path='/vulcan/scratch/mtang/datasets/ABIDE/fmri/'
		self.subjects=df['FILE_ID']
		self.labels=df['DX_GROUP']
		self.transform=transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self,subID):
		name=self.subjects.iloc[subID]
		img_name=self.path+self.subjects.iloc[subID]+'.nii.gz'

		img=nib.load(img_name)
		label=self.labels[subID]-1

        # save img data array
		imgData=img.get_fdata()
		newImg=torch.Tensor(imgData)
		newImg=newImg.permute(3,0,1,2)

		sample = [newImg, label]
		if self.transform:
			sample = self.transform(sample)
        
		return sample

