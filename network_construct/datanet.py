import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# roi 1d-conv net
class conv1D(nn.Module):
    def __init__(self, num_classes):
        super(conv1D, self).__init__()
        self.conv1 = nn.Conv1d(116, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        # self.conv3 = nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn3 = nn.BatchNorm1d(64)
        # self.conv4 = nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn4 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, num_classes)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        # x = F.relu(self.conv3(x))
        # x = self.bn3(x)
        # x = F.relu(self.conv4(x))
        # x = self.bn4(x)
        x = self.pool(x)
        x = self.fc(x.squeeze(2))
        return x

# fmri 3d-conv net
class conv3D(nn.Module):
    def __init__(self, num_classes):
        super(conv3D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(116, 64, kernel_size=(1,5,5), stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=1),

            nn.Conv3d(64, 192, kernel_size=(1,5,5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=1),

            nn.Conv3d(192, 384, kernel_size=(1,3,3), padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(384, 256, kernel_size=(1,3,3), padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 32, kernel_size=(1,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32*22*13*16, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32*22*13*16)
        x = self.classifier(x)
        return x

class complete_data(nn.Module):
    def __init__(self, num_classes):
        super(complete_data, self).__init__()

        fmodel=conv3D(num_classes=num_classes)
        self.fmri=nn.Sequential(*list(fmodel.children())[:-1])

        rmodel=conv1D(num_classes=num_classes)
        self.roi=nn.Sequential(*list(rmodel.children())[:-2])

        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear((32*22*13*16)+(64*8), 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, f, r):
        f=self.fmri(f)
        f=f.view(f.size(0), 32*22*13*16)

        r=self.roi(r)
        r=r.view(r.size(0), 64*8)

        x=torch.cat((f,r),1)        
        x=self.classifier(x)

        return x

