import nibabel as nib
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

listID=os.listdir('vulcan/scratch/mtang/datasets/ABIDE/min_process')

df = pd.read_excel('abide_summary.xlsx', sheet_name='summary')
classLabel=df['DX_GROUP']

for i in range(1, len(listID)):
	idName=str(listID[i])
	img=nib.load('/vulcan/scratch/mtang/datasets/ABIDE/min_process/'+idName+'.nii.gz')
	imgData=img.get_fdata()
    size=imgData.shape
    for i in range(0,shape[3]):
        slice=img[:,:,i,1]
        print('slice: '+str(slice))
