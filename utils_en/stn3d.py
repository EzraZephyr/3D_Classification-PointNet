import torch
from torch import nn


class STN3D(nn.Module):
    def __init__(self):
        super(STN3D, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        # Extract features from x, y, z coordinates and map them to a 3x3 matrix
        # This improves alignment of input point clouds from different angles and positions, making the model robust to transformations like rotations and translations

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        # Extract the maximum value across all points for each feature dimension

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, device=x.device).view(1, 9).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        # Since weights are initialized randomly,
        # adding an identity matrix helps prevent large deviations in the initial predicted transformation matrix
        # This ensures the transformation matrix is close to the identity matrix in the early stages, stabilizing training
        # As training progresses, the model learns meaningful features,
        # and the identity matrix has less influence, avoiding negative impact on results

        return x
