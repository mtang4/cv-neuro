import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from fmri import fmri
class maps(Dataset):
    """Loads connectivity maps as pre-saved tensors."""

    def __init__(self,transform=None):
        df=pd.read_excel('/vulcan/scratch/mtang/code/neuroimaging/preprocessing/abide_summary.xlsx', sheet_name='summary')
        self.path='/vulcan/scratch/mtang/datasets/ABIDE/connect/'
        self.allsubjects=list(df['FILE_ID'])

    def __len__(self):
        return len(self.allsubjects)

    def __getitem__(self, subID):
        name=self.allsubjects[subID]
        [matrix, label]=torch.load(self.path+name+'.pkl')
        matrix=matrix.numpy()
        matrix=np.nan_to_num(matrix)
        return [matrix, label]


class maps_1d(Dataset):
    """Loads ROI time series data as pre-saved tensors."""

    def __init__(self,transform=None):
        path = '/vulcan/scratch/mtang/code/neuroimaging'
        data_file = 'data_1d.pkl'
        label_file = 'label_1d.pkl'
        with open(os.path.join(path, data_file), 'rb') as f:
            self.data = pickle.load(f)
        with open(os.path.join(path, label_file), 'rb') as f:
            self.label = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, subID):
        img = self.data[subID]
        length = img.shape[0]
        num_segments = 116
        unit_len = length / (num_segments + 1)
        offsets = np.multiply(np.arange(num_segments) + np.random.random(num_segments), unit_len) 
        offsets = offsets.round().astype(np.int)
        label = self.label[subID]
        return [img[offsets], label]