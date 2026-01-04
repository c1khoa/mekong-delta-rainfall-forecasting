import os
import numpy as np
import torch
from torch.utils.data import Dataset


class FolderTimeSeriesDataset(Dataset):
    """Dataset cho chuỗi thời gian lưu theo folder (.npy)."""

    def __init__(self, root_dir, time_in=30, time_out=1):
        """
        root_dir: ví dụ 'data/test' (tính từ thư mục demo/)
        """
        self.feature_dir = os.path.join(root_dir, "features")
        self.target_dir  = self.feature_dir

        self.feature_files = sorted(os.listdir(self.feature_dir))
        self.target_files  = sorted(os.listdir(self.target_dir))
        assert len(self.feature_files) == len(self.target_files), "features != target"

        self.time_in = time_in
        self.time_out = time_out

        self.sample_indices = []
        total_timesteps = len(self.feature_files)
        for i in range(total_timesteps - time_in - time_out + 1):
            self.sample_indices.append(i)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_idx = self.sample_indices[idx]

        X = np.stack([
            np.transpose(np.load(os.path.join(self.feature_dir, self.feature_files[i])), (2,0,1))
            for i in range(start_idx, start_idx + self.time_in)
        ], axis=0)

        Y = np.stack([
            np.transpose(np.load(os.path.join(self.target_dir, self.target_files[i])), (2,0,1))
            for i in range(start_idx + self.time_in, start_idx + self.time_in + self.time_out)
        ], axis=0)

        return torch.FloatTensor(X), torch.FloatTensor(Y)


class TensorTimeSeriesDataset(Dataset):
    """Dataset dạng tensor 3D time-series lưu trên disk."""

    def __init__(self, root_dir):
        self.feature_dir = os.path.join(root_dir, "features")
        self.target_dir = os.path.join(root_dir, "target")
        self.feature_files = sorted(os.listdir(self.feature_dir))
        self.target_files = sorted(os.listdir(self.target_dir))

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        feat = np.load(os.path.join(self.feature_dir, self.feature_files[idx]))
        targ = np.load(os.path.join(self.target_dir, self.target_files[idx]))
        feat = torch.from_numpy(feat).float().permute(2,0,1)
        targ = torch.from_numpy(targ).float().permute(2,0,1)
        return feat, targ
