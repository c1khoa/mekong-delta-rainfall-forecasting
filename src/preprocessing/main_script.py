import warnings
import os
import torch
import joblib

from config import Config
from pipeline import MekongDataPipeline
from utils import minmax_scale_tensor, save_timeseries_tensor

warnings.filterwarnings('ignore')


def main():
    config = Config(
        output_dir="../../data/preprocessed",
        chunk_size=50,
        use_multiprocessing=True,
        n_workers=None
    )

    os.makedirs(config.output_dir, exist_ok=True)

    pipeline = MekongDataPipeline(config)
    results = pipeline.run()
    joblib.dump(pipeline.processor.robust_scaler_target, config.robust_scaler_path)

    tensors = results['tensors']
    print("\nTensor shapes:")
    for name, tensor in tensors.items():
        print(f"{name:20s}: {tensor.shape}")

    train_data = torch.cat((
        torch.from_numpy(tensors['mekong_train']).float(),
        torch.from_numpy(tensors['sentinel_1_train']).float(),
        torch.from_numpy(tensors['sentinel_2_train']).float()
    ), dim=-1)

    val_data = torch.cat((
        torch.from_numpy(tensors['mekong_val']).float(),
        torch.from_numpy(tensors['sentinel_1_val']).float(),
        torch.from_numpy(tensors['sentinel_2_val']).float()
    ), dim=-1)

    test_data = torch.cat((
        torch.from_numpy(tensors['mekong_test']).float(),
        torch.from_numpy(tensors['sentinel_1_test']).float(),
        torch.from_numpy(tensors['sentinel_2_test']).float()
    ), dim=-1)

    train_target = torch.from_numpy(tensors['target_train'])
    val_target = torch.from_numpy(tensors['target_val'])
    test_target = torch.from_numpy(tensors['target_test'])

    train_data_scale, val_data_scale, test_data_scale = minmax_scale_tensor(
        train_data, val_data, test_data
    )

    train_target_scale, val_target_scale, test_target_scale = minmax_scale_tensor(
        train_target, val_target, test_target, scaler_path=config.minmax_scaler_path
    )

    save_timeseries_tensor(
        train_data_scale, val_data_scale, test_data_scale,
        train_target_scale, val_target_scale, test_target_scale,
        base_path=config.dataset_ts_path
    )


if __name__ == "__main__":
    main()