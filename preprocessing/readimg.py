import nibabel as nib
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

sites=['caltech', 'CMU', 'KKI', 'leuven', 'maxmun', 'NYU', 'OHSU', 'olin',
       'pitt', 'SBL', 'SDSU', 'stanford', 'trinity', 'UCLA', 'UM', 'USM',
       'yale']

# loop through sites
for i in range(13,17):
    siteName=sites[i]
    print('now checking: '+siteName)
    df = pd.read_excel('ABIDE.xlsx', sheet_name=siteName)
    listID = df['Subject']

    # loop through subjects
    for i in df.index:
        idName=str(df['Subject'][i])
        img=nib.load('/Volumes/FREEAGENT D/umd 2019 data/neuroimaging/ABIDE/'+
                     'ANTs/'+siteName+'/'+idName+'/'+idName+'-ants/ants/'+
                     'anat_thickness.nii.gz')
        imgData=img.get_fdata()
        if i==0:
            initialDim=imgData.shape
            print('initial dim: '+str(initialDim))
        else:
            if imgData.shape!=initialDim:
                print('FLAG: subject '+idName)
                
