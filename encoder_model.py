import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self,feature_dim):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=32,stride=5,padding=16)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3,stride=1,padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=3,stride=1,padding=1)
        self.bn5 = nn.BatchNorm1d(128)


        self.fc1 = nn.Linear(1536, feature_dim)
        self.bn6 = nn.BatchNorm1d(feature_dim)
        self.fc2 = nn.Linear(feature_dim,10)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out=F.normalize(out)
        out=self.relu(out)
        out = self.dropout(out)
        fea = out
        return fea, out
