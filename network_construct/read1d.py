import os
import numpy as np
import torch
import pickle
import pandas as pd


np.seterr(divide='ignore', invalid='ignore')
rootpath='/vulcan/scratch/mtang/datasets/ABIDE/ROI/'
df=pd.read_excel('/vulcan/scratch/mtang/code/neuroimaging/preprocessing/abide_summary.xlsx', sheet_name='summary')
savepath='/vulcan/scratch/mtang/datasets/ABIDE/connect/'
allsubjects=list(df['FILE_ID'])
labels=list(df['DX_GROUP'])


for i in range(len(allsubjects)):
    subject=allsubjects[i]
    print(subject)
    allvoxels=[]
    linenum=0
    label=int(labels[i])-1

    with open(rootpath+subject+'.1D') as f:

        # read data
        for line in f:
            data=line.split('\t')
            if linenum>0:
                data=[float(i) for i in data]
                allvoxels.append(data)
            linenum+=1
        
        # compute matrix
        allvoxels=np.array(allvoxels)
        allvoxels=allvoxels.transpose()
        connectivity=np.corrcoef(allvoxels)
        connectivity=np.triu(connectivity,1)
        connectivity=connectivity[connectivity!=0]

        connectivity=torch.Tensor(connectivity)

        #with open(savepath+subject+'.pkl', 'wb') as m:
            #pickle.dump([subject,connectivity,label], m)
