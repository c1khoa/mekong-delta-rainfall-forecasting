import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


def minmax_scale_tensor(train, val, test, scaler_path=None):
    n_channels = train.shape[-1]
    train_scaled = np.zeros_like(train, dtype=np.float32)
    val_scaled   = np.zeros_like(val, dtype=np.float32)
    test_scaled  = np.zeros_like(test, dtype=np.float32)

    for c in range(n_channels):
        scaler = MinMaxScaler()

        train_flat = train[..., c].reshape(-1, 1)
        val_flat   = val[..., c].reshape(-1, 1)
        test_flat  = test[..., c].reshape(-1, 1)

        scaler.fit(train_flat)

        train_scaled[..., c] = scaler.transform(train_flat).reshape(train.shape[0], train.shape[1], train.shape[2])
        val_scaled[..., c]   = scaler.transform(val_flat).reshape(val.shape[0], val.shape[1], val.shape[2])
        test_scaled[..., c]  = scaler.transform(test_flat).reshape(test.shape[0], test.shape[1], test.shape[2])

        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)

    return train_scaled, val_scaled, test_scaled


def save_timeseries_tensor(train_data, val_data, test_data,
                           train_target, val_target, test_target,
                           base_path):
    sets = {
        "train": (train_data, train_target),
        "val":   (val_data, val_target),
        "test":  (test_data, test_target)
    }

    for set_name, (data, target) in sets.items():
        feature_dir = os.path.join(base_path, set_name, "features")
        target_dir  = os.path.join(base_path, set_name, "target")
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)

        n_samples = data.shape[0]

        for i in range(n_samples):
            feat_path = os.path.join(feature_dir, f"{i:04d}.npy")
            targ_path = os.path.join(target_dir,  f"{i:04d}.npy")

            np.save(feat_path, data[i])
            np.save(targ_path, target[i])

    print(f"Lưu xong dataset time-series vào {base_path}!")