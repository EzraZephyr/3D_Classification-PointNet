import torch
from stn3d import STN3D
from torch import nn

class PointNet(nn.Module):
    def __init__(self, npf, num_class=40):
        super(PointNet, self).__init__()
        self.stn = STN3D()
        self.conv1 = nn.Conv1d(3, npf, 1)
        self.conv2 = nn.Conv1d(npf, npf, 1)

        self.conv3 = nn.Conv1d(npf, npf, 1)
        self.conv4 = nn.Conv1d(npf, npf*2, 1)
        self.conv5 = nn.Conv1d(npf*2, npf*16, 1)

        self.fc1 = nn.Linear(npf*16, npf*8)
        self.fc2 = nn.Linear(npf*8, npf*4)
        self.fc3 = nn.Linear(npf*4, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(npf)
        self.bn2 = nn.BatchNorm1d(npf)
        self.bn3 = nn.BatchNorm1d(npf)
        self.bn4 = nn.BatchNorm1d(npf*2)
        self.bn5 = nn.BatchNorm1d(npf*16)
        self.bn6 = nn.BatchNorm1d(npf*8)
        self.bn7 = nn.BatchNorm1d(npf*4)

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        # Transpose input point cloud to shape (batch_size, num_points, 3) to align with STN's transformation matrix
        # This allows matrix multiplication to apply the transformation to each point correctly

        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))

        x = torch.max(x, 2)[0]
        # Extract the maximum value across all points for each feature dimension

        x = self.relu(self.bn6(self.fc1(x)))
        x = self.relu(self.bn7(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return x
