import torch
from torch import nn


class STN3D(nn.Module):
    def __init__(self):
        super(STN3D, self).__init__()
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        # 将xyz的坐标分别进行特征提取 最后全部映射为3*3的矩阵
        # 这样可提高对不同视角和位置输入点云的对齐能力 以鲁棒点云的旋转平移等操作

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
        x = torch.max(x,2)[0]
        # 提取所有点中每个特征维度影响最大的值

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, device = x.device).view(1, 9).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        # 因为权重是随机初始化的 为了防止模型在训练初期的预测变换矩阵出现过大偏移而导致模型崩溃
        # 需要初始化单位矩阵 以加上预测的偏移 确保最开始几次训练的变化接近单位矩阵
        # 再训练增加后 模型开始自己学到特征 单位矩阵的作用将会越来越小 不会对模型结果产生负面影响

        return x
