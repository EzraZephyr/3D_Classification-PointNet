import os
import h5py
import numpy as np
from torch.utils.data import Dataset

def load_h5data(filename):
    with h5py.File(filename, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]
        # Extract data and labels from the specified file

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
            # Choose to load either the training or test set and build the complete path

        with open(file_path, 'r') as f:
            self.h5_files = [os.path.join(self.data_dir, line.strip().split('/')[-1]) for line in f]
            # Read each line, extract the last part of the filename, and build a complete path

        self.data, self.labels = self.load_data()

    def load_data(self):
        all_data = []
        all_labels = []
        for h5_file in self.h5_files:
            data, label = load_h5data(h5_file)
            all_data.append(data)
            all_labels.append(label)
            # Iterate over each path in the collection, pass it to load_h5data, and store the extracted data and labels

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        # Concatenate all samples along the first dimension; final shape is (n_total, 2048, 3) and (i_total, 1)

        return all_data, all_labels.squeeze()
        # Remove the extra dimension of size 1 from labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        point = self.data[idx]
        choice = np.random.choice(point.shape[0], self.num_points, replace=False)
        point = point[choice, :]
        # Randomly sample 1024 points from the 2048 points
        # Not using farthest point sampling here because it's too computationally intensive, GPU can't handle it :(

        label = self.labels[idx]
        return point, label
