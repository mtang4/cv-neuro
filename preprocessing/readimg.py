import nibabel as nib
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

import os

# sites=['caltech', 'CMU', 'KKI', 'leuven', 'maxmun', 'NYU', 'OHSU', 'olin',
#       'pitt', 'SBL', 'SDSU', 'stanford', 'trinity', 'UCLA', 'UM', 'USM',
#       'yale']

df = pd.read_excel('ABIDE_Phenotypics.xlsx', sheet_name='5320_ABIDE_Phenotypics_20190625')
listID=df['Anonymized ID']

for i in df.index:
       idName=str(df['Subject'][i])
       dirlist = os.listdir('/vulcan/scratch/mtang/datasets/ABIDE/allSubjects'+idName)
       subFolder=str(dirlist[0])
       img=nib.load('/vulcan/scratch/mtang/datasets/ABIDE/allSubjects'+idName+subFolder+'rest_0001/REST.nii.gz')
       imgData=img.get_fdata()
       if i==0:
            initialDim=imgData.shape
            print('initial dim: '+str(initialDim))
        else:
            if imgData.shape!=initialDim:
                print('FLAG: subject '+idName)
  
