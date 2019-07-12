import nibabel as nib
import os

filePath='/vulcan/scratch/mtang/datasets/ABIDE/min_process'
fileList=os.listdir(filePath)

for i in range(0, 1035):
    img=nib.load(filePath+'/'+fileList[i])
    imgData=img.get_fdata()
    print('subject is: '+str(fileList[i])+' - '+str(imgData.shape))
