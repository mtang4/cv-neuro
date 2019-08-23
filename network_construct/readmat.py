import scipy.io as sio
import pickle
import os
import numpy as np
import torch

inpath='/vulcan/scratch/mtang/datasets/ABIDE/fmri/corrmatrix/'
outpath='/vulcan/scratch/mtang/datasets/ABIDE/fmri/inputs/'


files=os.listdir(inpath)

for i in range(len(files)):
    filename='subject'+str(i+1)+'.mat'
    print(filename+'\n')

    m=sio.loadmat(inpath+filename)
    label=int(list(m.values())[3])
    r=list(m.values())[4]
    
    # clean up and reshape data
    r=np.nan_to_num(r)
    r=np.reshape(r, (61,73,61,116))
    r_tensor=torch.Tensor(r)
    r_tensor=r_tensor.permute(3,0,1,2)
    data=[r_tensor, label]

    # save data
    with open(outpath+'subject'+str(i+1)+'.pkl', 'wb') as f:
        pickle.dump(data, f)