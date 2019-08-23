import torch.nn as nn


class SimpleConv(nn.Module):
    def __init__(self, num_classes):
        super(SimpleConv, self).__init__()

        # feature extraction: 5 layers
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

        # classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(32*22*13*16, 4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32*22*13*16)
        x = self.classifier(x)
        return x


def simple_conv(pretrained=False, num_classes=2):
    model = SimpleConv(num_classes)
    return model