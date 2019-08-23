import os
import pickle
import torch
from torch.utils.data import Dataset

class corrmat(Dataset):
    """Loads correlation matrices as pre-saved tensors."""

    def __init__(self,transform=None):
        self.path = '/vulcan/scratch/mtang/datasets/ABIDE/inputs/'
        self.files=os.listdir(self.path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, subID):
        with open(self.path+self.files[subID], 'rb') as f:
            item=pickle.load(f)
        return item



