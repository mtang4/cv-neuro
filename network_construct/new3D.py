import torch.nn as nn


class NewConv(nn.Module):
    def __init__(self, num_classes):
        super(NewConv, self).__init__()

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
        # print(str(x.size(0))+', '+str(x.size(1))+', '+str(x.size(2))+', '+str(x.size(3))+', '+str(x.size(4)))
        x = x.view(x.size(0), 32*15*18*15)      # flatten
        x = self.classifier(x)
        return x


def new_conv(pretrained=False, num_classes=2):
    model = NewConv(num_classes)
    return model