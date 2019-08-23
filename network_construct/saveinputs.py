from __future__ import print_function
import argparse

import numpy as np
import os
import scipy.io as sio
import pickle

from cpac import cpac
from transforms import MRISelect
from fmri import fmri


# ignore warnings
import warnings
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

# path='/vulcan/scratch/mtang/datasets/ABIDE/corrmatrix/tensors/'
# allData=fmri()

path='/vulcan/scratch/mtang/datasets/ABIDE/cpac/tensors/'
allData=cpac(transform=MRISelect())

"""
for i in range(len(allData)):
        print('subject: '+str(i+1))
        filename=path+'subject'+str(i+1)+'.mat'
        item=allData[i]
        sio.savemat(filename,mdict={'img':item[0], 'roi':item[1], 'label':item[2]})
"""

for i in range(len(allData)):
        print('subject: '+str(i+1))
        filename=path+'subject'+str(i+1)
        data=allData[i]
        with open(filename+'.pkl', 'wb') as f:
                pickle.dump(data, f)