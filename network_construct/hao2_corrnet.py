import torch
import os
import torch.nn as nn
import torch.nn.functional as F

# from IPython import embed;embed()
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
            # nn.AvgPool3d(kernel_size=(2,2,2)),
            nn.AdaptiveAvgPool3d((32,32,32)),
            nn.Conv3d(61, 128, kernel_size=(3,3,3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=2, padding=1),
            nn.ELU(),
            # nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=1),
            # nn.ELU(),
            # nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=2, padding=1),
            # nn.ELU(),
            # nn.Conv3d(512, 512, kernel_size=(3,3,3), stride=1, padding=1),
            # nn.ELU(),
            # nn.Conv3d(512, 1024, kernel_size=(3,3,3), stride=2, padding=1),
            # nn.ELU(),

            # # nn.MaxPool3d(kernel_size=(2,2,2)),
            nn.AdaptiveMaxPool3d(3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256*3*3*3, 32),
            nn.ELU(),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*3*3*3)      
        x = self.classifier(x)
        return x

### classifiers
class small_class(nn.Module):
    def __init__(self, num_classes):
        super(small_class, self).__init__()

        # feature extractors
        fmodel=fmrixroi(num_classes=num_classes)
        self.fmri=nn.Sequential(*list(fmodel.children())[:-1])

        rmodel=roixroi(num_classes=num_classes)
        self.roi=nn.Sequential(*list(rmodel.children())[:-1])

        
        # fully connected classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear((1024*3*3*3)+100, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        

    def forward(self, f, r):
        f=self.fmri(f)
        f=f.view(f.size(0), 256*3*3*3)

        r=self.roi(r)

        x=torch.cat((f,r),1)
        x=self.classifier(x)

        return x

class large_class(nn.Module):
    def __init__(self, num_classes):
        super(large_class, self).__init__()

        # feature extractors
        fmodel=fmrixroi(num_classes=num_classes)
        self.fmri=nn.Sequential(*list(fmodel.children())[:-1])

        rmodel=roixroi(num_classes=num_classes)
        self.roi=nn.Sequential(*list(rmodel.children())[:-1])

        # fully connected classifier
        self.classifier = nn.Sequential(
            # nn.Linear((256*3*3*3)+100, 4096),
            # nn.Linear(100, 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, num_classes)
            nn.Linear(100, num_classes)
        )

    def forward(self, fc_data, fmri_data):
        # import pdb;pdb.set_trace()
        # f=self.fmri(fmri_data.permute(0,4,1,2,3))
        # f=f.view(f.size(0), 256*3*3*3)

        r=self.roi(fc_data)
        x=self.classifier(r)

        # x=torch.cat((f,r),1)
        # x=self.classifier(x)

        # x=self.classifier(x)

        return x
        
