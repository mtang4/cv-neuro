import nibabel as nib
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

import os

df = pd.read_excel('abide_summary.xlsx', sheet_name='abide_summary')
listID=df['FILE_ID']
classLabel=df['DX_GROUP']

countImg=0
for i in range(1, len(listID)):
	idName=str(listID[i])
	# dirlist = os.listdir('/vulcan/scratch/mtang/datasets/ABIDE/min_process/'+idName)
	# subFolder=str(dirlist[0])
	img=nib.load('/vulcan/scratch/mtang/datasets/ABIDE/min_process/'+idName+'.nii.gz')
	countImg+=1
	imgData=img.get_fdata()
	if i==1:
		initialDim=imgData.shape
		print('dim: '+str(initialDim))
	else:
		if imgData.shape!=initialDim:
			newDim=imgData.shape
			print('subject '+idName+' - new dim: '+str(imgData.shape))
			initialDim=newDim
  
print('total images: '+count)
