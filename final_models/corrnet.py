import torch
import os
import torch.nn as nn
import torch.nn.functional as F


DROP_RATIO=.5

class roixroi(nn.Module):
    def __init__(self, num_classes):
        super(roixroi, self).__init__()
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(6670, 100)
        self.drop2 = nn.Dropout(p=DROP_RATIO)
        self.fc2 = nn.Linear(100, 100)
        self.drop3 = nn.Dropout(p=DROP_RATIO)
        self.fc3 = nn.Linear(100, 100)
        self.drop4 = nn.Dropout(p=DROP_RATIO)
        self.fc4 = nn.Linear(100, 100)
        self.drop5 = nn.Dropout(p=DROP_RATIO)
        self.fc5 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = F.relu(self.fc3(x))
        x = self.drop4(x)
        x = F.relu(self.fc4(x))
        x = self.drop5(x)
        x = self.fc5(x)
        return x


class fmrixroi(nn.Module):
    def __init__(self, num_classes):
        super(fmrixroi, self).__init__()

        # define network layers (as according to CNN paper)
        self.features = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2,2,2)),

            nn.Conv3d(116, 128, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ELU(),

            nn.Conv3d(128, 32, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ELU(),

            nn.MaxPool3d(kernel_size=(2,2,2)),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32*15*18*15, 32),
            nn.ELU(),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32*15*18*15)      
        x = self.classifier(x)
        return x

### classifiers
class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()

        # feature extractors
        rmodel=torch.load('/vulcan/scratch/mtang/code/neuroimaging/network_construct/roixroi.pt')
        self.roi=nn.Sequential(*list(rmodel.children())[:-1])

        fmodel=torch.load('/vulcan/scratch/mtang/code/neuroimaging/network_construct/fmrixroi.pt')
        self.fmri=nn.Sequential(*list(fmodel.children())[:-1])
        self.addlayer=list(fmodel.children())[-1][0]
        

        # fully connected classifier
        self.fcnet = nn.Sequential(
            nn.Linear(32+100, 4096),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, r, f):
        r=self.roi(r)

        f=self.fmri(f)
        f=f.view(f.size(0), 32*15*18*15)
        f=self.addlayer(f)

        x=torch.cat((f,r),1)
        x=self.fcnet(x)

        return x
        
