import os
import torch
import random
from torch.utils.data import Dataset
import numpy as np

class corrdata(Dataset):
    """Loads correlation matrices as pre-saved tensors."""

    def __init__(self, list_file, load_fmri=True, transform=None):
        with open(list_file) as f:
            lines=f.readlines()
        self.img_list = [ele.strip() for ele in lines]
        self.fc_dir = '/vulcan/scratch/mtang/datasets/ABIDE/connect'
        self.fmri_dir = '/vulcan/scratch/mtang/datasets/ABIDE/hao_inputs'
        self.load_fmri = load_fmri 

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, subID):
        img_name = self.img_list[subID]
        with open(os.path.join(self.fc_dir, img_name+'.pkl'), 'rb') as f:
            fc_data, fc_label = torch.load(f)
            fc_data = np.nan_to_num(fc_data)
            if not self.load_fmri:
                return fc_data, fc_label
        with open(os.path.join(self.fmri_dir, img_name+'.p'), 'rb') as f:
            fmri_data, fmri_label = torch.load(f)
        assert fc_label == fmri_label
        return [fc_data, fmri_data.float(), fc_label]
