import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FolderTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset load từng file .npy từ folder:
    dataset_ts/train/val/test
        features/0000.npy
        target/0000.npy
    """
    def __init__(self, root_dir, time_in=30, time_out=1):
        self.feature_dir = os.path.join(root_dir, "features")
        self.target_dir  = self.feature_dir

        self.feature_files = sorted(os.listdir(self.feature_dir))
        self.target_files  = sorted(os.listdir(self.target_dir))
        assert len(self.feature_files) == len(self.target_files), "features != target"

        self.time_in = time_in
        self.time_out = time_out

        # Tạo danh sách index cho sliding window
        self.sample_indices = []
        total_timesteps = len(self.feature_files)
        for i in range(total_timesteps - time_in - time_out + 1):
            self.sample_indices.append(i)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_idx = self.sample_indices[idx]

        # Load input sequence
        X_list = []
        for i in range(start_idx, start_idx + self.time_in):
            feat = np.load(os.path.join(self.feature_dir, self.feature_files[i]))  # [H,W,C]
            feat = np.transpose(feat, (2,0,1))  # [C,H,W]
            X_list.append(feat)
        X = np.stack(X_list, axis=0)  # [time_in, C, H, W]

        # Load output sequence
        Y_list = []
        for i in range(start_idx + self.time_in, start_idx + self.time_in + self.time_out):
            targ = np.load(os.path.join(self.target_dir, self.target_files[i]))  # [H,W,C] (C=1)
            targ = np.transpose(targ, (2,0,1))  # [C,H,W]
            Y_list.append(targ)
        Y = np.stack(Y_list, axis=0)  # [time_out, C, H, W]

        return torch.FloatTensor(X), torch.FloatTensor(Y)

class TensorTimeSeriesDataset(Dataset):
    """
    Dataset PyTorch cho tensor 3D time-series lưu trên disk
    """
    def __init__(self, root_dir):
        """
        root_dir: đường dẫn tới folder 'train', 'val' hoặc 'test'
        """
        self.feature_dir = os.path.join(root_dir, "features")
        self.target_dir  = os.path.join(root_dir, "target")

        self.feature_files = sorted(os.listdir(self.feature_dir))
        self.target_files  = sorted(os.listdir(self.target_dir))

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        # Load feature và target
        feat_path = os.path.join(self.feature_dir, self.feature_files[idx])
        targ_path = os.path.join(self.target_dir, self.target_files[idx])

        feature = np.load(feat_path)  # [H, W, C]
        target  = np.load(targ_path)  # [H, W, 1]

        # Chuyển sang tensor và permute để Conv2D input: [C, H, W]
        feature = torch.from_numpy(feature).float().permute(2, 0, 1)  # [C,H,W]
        target  = torch.from_numpy(target).float().permute(2, 0, 1)   # [1,H,W]

        return feature, target

def get_dataloaders(
    path,
    time_in=30,
    time_out=14,
    batch_size_1=8,
    batch_size_2=16,
    num_workers=4,
    pin_memory=True,
    drop_last=False
):
    """
    Trả về:
    {
        "stage_1": {
            "train": <DataLoader>,
            "val":   <DataLoader>,
            "test":  <DataLoader>
        },
        "stage_2": {
            "train": <DataLoader>,
            "val":   <DataLoader>,
            "test":  <DataLoader>
        }
    }
    """
    result = {
        "stage_1": {},
        "stage_2": {}
    }

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(path, split)

        # 1) FolderTimeSeriesDataset (sliding window)
        folder_ds = FolderTimeSeriesDataset(
            root_dir=split_dir,
            time_in=time_in,
            time_out=time_out
        )
        result["stage_1"][split] = DataLoader(
            folder_ds,
            batch_size=batch_size_1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )

        # 2) TensorTimeSeriesDataset (từng timestep)
        tensor_ds = TensorTimeSeriesDataset(split_dir)
        result["stage_2"][split] = DataLoader(
            tensor_ds,
            batch_size=batch_size_2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    folder_train = result["stage_1"]["train"]
    tensor_train = result["stage_2"]["train"]
    x1, y1 = next(iter(folder_train))
    x2, y2 = next(iter(tensor_train))
    
    print("Folder batch train (model 1):", "Input: ", x1.shape, "Output", y1.shape)
    print("Folder batch train (model 2):", "Input: ", x2.shape, "Output", y2.shape)

    return result