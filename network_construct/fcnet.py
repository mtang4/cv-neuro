import torch.nn as nn
import torch.nn.functional as F

DROP_RATIO=.5

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
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


class conv1d(nn.Module):
    def __init__(self, num_classes):
        super(conv1d, self).__init__()
        self.conv1 = nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, num_classes)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool(x)
        x = self.fc(x.squeeze(2))
        return x
