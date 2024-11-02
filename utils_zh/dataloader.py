import os
import h5py
import numpy as np
from torch.utils.data import Dataset

def load_h5data(filename):
    with h5py.File(filename, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]
        # 取出文件名下的数据和标签

    f.close()
    return data, label

class ModelNet40Dataset(Dataset):
    def __init__(self, data_dir, split='train', num_points=1024):
        self.data_dir = data_dir
        self.split = split
        self.num_points = num_points

        if self.split == 'train':
            file_path = os.path.join(self.data_dir, 'train_files.txt')
        else:
            file_path = os.path.join(self.data_dir, 'test_files.txt')
            # 选择加载训练集还是测试集 并拼成一个完整的路径

        with open(file_path, 'r') as f:
            self.h5_files = [os.path.join(self.data_dir, line.strip().split('/')[-1]) for line in f]
            # 逐行读取并提取最后一个的文件名并分别组合成完整的路径

        self.data, self.labels = self.load_data()

    def load_data(self):
        all_data = []
        all_labels = []
        for h5_file in self.h5_files:
            data, label = load_h5data(h5_file)
            all_data.append(data)
            all_labels.append(label)
            # 遍历路径集合的每一个路径 传入load_h5data中提取出数据和标签进行储存

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        # 将所有样本延第一维度拼接起来 最后形状为(n_total, 2048, 3)和(i_total, 1)

        return all_data, all_labels.squeeze()
        # 取出标签的多余维度1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        point = self.data[idx]
        choice = np.random.choice(point.shape[0], self.num_points, replace=False)
        point = point[choice, :]
        # 随机从2048个点中采样1024个点
        # 这里没有用最远点采样的原因是因为太消耗算力了 GPU扛不住:(

        label = self.labels[idx]
        return point, label
