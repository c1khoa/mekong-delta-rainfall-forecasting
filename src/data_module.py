import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PipelineTimeSeriesDataset(Dataset):
    """
    Dataset chuẩn cho pipeline ST-model -> Decoder
    Trả về input sequence (time_in) và target multi-step (time_out)
    """
    def __init__(self, root_dir, time_in=30, time_out=14):
        self.feature_dir = os.path.join(root_dir, "features")
        self.target_dir  = os.path.join(root_dir, "target")

        self.feature_files = sorted(os.listdir(self.feature_dir))
        self.target_files  = sorted(os.listdir(self.target_dir))
        assert len(self.feature_files) == len(self.target_files), "Features và target không khớp"

        self.time_in = time_in
        self.time_out = time_out

        # Chuẩn bị các index sliding window
        self.sample_indices = []
        total_timesteps = len(self.feature_files)
        for i in range(total_timesteps - time_in - time_out + 1):
            self.sample_indices.append(i)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_idx = self.sample_indices[idx]

        # Input sequence cho ST-model
        X_list = []
        for i in range(start_idx, start_idx + self.time_in):
            feat = np.load(os.path.join(self.feature_dir, self.feature_files[i]))
            feat = np.transpose(feat, (2,0,1))  # (C, H, W)
            X_list.append(feat)
        X = np.stack(X_list, axis=0)  # (time_in, C, H, W)

        # Target multi-step cho Decoder
        Y_list = []
        for i in range(start_idx + self.time_in, start_idx + self.time_in + self.time_out):
            targ = np.load(os.path.join(self.target_dir, self.target_files[i]))
            targ = np.transpose(targ, (2,0,1))  # (C, H, W)
            Y_list.append(targ)
        Y = np.stack(Y_list, axis=0)  # (time_out, C, H, W)

        return torch.FloatTensor(X), torch.FloatTensor(Y)


def get_dataloaders(
    path,
    time_in=30,
    time_out=14,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    drop_last=False
):
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = PipelineTimeSeriesDataset(
            root_dir=os.path.join(path, split),
            time_in=time_in,
            time_out=time_out
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if split=="train" else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        loaders[split] = loader

    return loaders

loaders = get_dataloaders("../data/preprocessed")

batch_x, batch_y = next(iter(loaders["train"]))
print("Batch X:", batch_x.shape)
print("Batch Y:", batch_y.shape)

