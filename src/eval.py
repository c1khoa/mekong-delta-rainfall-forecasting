import torch
import torch.nn as nn
import joblib
import json
import warnings
from itertools import product
from collections import defaultdict
from tqdm import tqdm
warnings.filterwarnings('ignore')

from models.convlstm import ConvLSTMForecaster
from models.unet import UNet
from models.convgru import ConvGRUForecaster
from models.fpn import RainfallFPN
from models.trajgru import TrajGRU
from models.unet3plus import UNet3Plus
from data_module import get_dataloaders

import config
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class FlexibleCombinedModel(nn.Module):
    def __init__(self, temporal_type, spatial_type, minmax_path, robust_path):
        super().__init__()
        self.temporal_type = temporal_type
        self.spatial_type = spatial_type
        
        self.temporal_model = self._build_temporal(temporal_type)
        self.spatial_model = self._build_spatial(spatial_type)
        
        self.minmax_scaler = joblib.load(minmax_path)
        self.robust_scaler = joblib.load(robust_path)
        
        print(f"  ✓ Built: {temporal_type} + {spatial_type}")
    
    def _build_temporal(self, temporal_type):
        if temporal_type == 'ConvLSTM':
            ckpt = torch.load(config.ConvLSTM.PATH, map_location=config.DEVICE, weights_only=True)
            state = ckpt["model_state_dict"]
            model = ConvLSTMForecaster(
                input_channels=config.ConvLSTM.INPUT_CHANNELS,
                hidden_dims=config.ConvLSTM.HIDDEN_DIMS,
                kernel_size=config.ConvLSTM.KERNEL_SIZE
            )
            model.load_state_dict(state, strict=False)
            return model
        
        elif temporal_type == 'ConvGRU':
            ckpt = torch.load(config.ConvGRU.PATH, map_location=config.DEVICE, weights_only=True)
            state = ckpt["model_state_dict"]
            model = ConvGRUForecaster(
                input_channels=config.ConvGRU.INPUT_CHANNELS,
                hidden_dims=config.ConvGRU.HIDDEN_DIMS,
                kernel_size=config.ConvGRU.KERNEL_SIZE
            )
            model.load_state_dict(state, strict=False)
            return model
        
        elif temporal_type == 'TrajGRU':
            ckpt = torch.load(config.TrajGRU.PATH, map_location=config.DEVICE, weights_only=True)
            state = ckpt["model_state_dict"]
            model = TrajGRU(
                time_out=config.TrajGRU.TIME_OUT,
                input_channels=config.TrajGRU.INPUT_CHANNELS,
                hidden_channels_list=config.TrajGRU.HIDDEN_DIMS,
                output_channels=config.TrajGRU.OUTPUT_CHANNELS,
                L=config.TrajGRU.L
            )
            model.load_state_dict(state, strict=False)
            return model
        
        else:
            raise ValueError(f"Unknown temporal type: {temporal_type}")
    
    def _build_spatial(self, spatial_type):
        if spatial_type == 'UNet':
            ckpt = torch.load(config.Unet.PATH, map_location=config.DEVICE, weights_only=True)
            state = ckpt["model_state_dict"]
            model = UNet(
                in_channels=config.Unet.INPUT_CHANNELS,
                out_channels=config.Unet.OUT_CHANNELS,
                features=config.Unet.FEATURES
            )
            model.load_state_dict(state, strict=False)
            return model
        
        elif spatial_type == 'FPN':
            ckpt = torch.load(config.FPN.PATH, map_location=config.DEVICE, weights_only=True)
            state = ckpt["model_state_dict"]
            model = RainfallFPN(
                in_channels=config.FPN.INPUT_CHANNELS,
                base_channels=config.FPN.BASE_CHANNELS,
                fpn_channels=config.FPN.FPN_CHANNELS
            )
            model.load_state_dict(state, strict=False)
            return model
        
        elif spatial_type == 'UNet3Plus':
            ckpt = torch.load(config.UNet3Plus.PATH, map_location=config.DEVICE, weights_only=True)
            state = ckpt["model_state_dict"]
            model = UNet3Plus(
                in_ch=config.UNet3Plus.INPUT_CHANNELS,
                out_ch=config.UNet3Plus.OUT_CHANNELS,
                base=config.UNet3Plus.BASE_CHANNELS
            )
            model.load_state_dict(state, strict=False)
            return model
        
        else:
            raise ValueError(f"Unknown spatial type: {spatial_type}")
    
    def forward(self, x):
        feat = self.temporal_model(x) # [B, time_out=1, C, H, W]
        feat = feat.squeeze(1) # [B, C, H, W]
        feat = self.spatial_model(feat) # [B, 1, H, W]
        
        feat = feat.detach().cpu().numpy()
        N, C, H, W = feat.shape
        feat = feat.reshape(-1, 1)
        feat = self.robust_scaler.inverse_transform(self.minmax_scaler.inverse_transform(feat)).reshape(N, C, H, W)
        
        return torch.from_numpy(feat).to(config.DEVICE)
    
def evaluate_rainfall(pred: torch.Tensor, target: torch.Tensor):
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)

    mae = mean_absolute_error(target_np, pred_np)
    rmse = mean_squared_error(target_np, pred_np, squared=False)
    r2 = r2_score(target_np, pred_np)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

test_loader = get_dataloaders(path=config.Dataset.PATH, 
                         time_in=config.Dataset.TIME_IN,
                         time_out=config.Dataset.TIME_OUT,
                         batch_size_1=config.Dataset.BATCH_SIZE_1,
                         batch_size_2=config.Dataset.BATCH_SIZE_2,
                         num_workers=config.Dataset.NUM_WORKERS,
                         pin_memory=config.Dataset.PIN_MEMORY)
stage_1 = test_loader['stage_1']["test"]
stage_2 = test_loader['stage_2']["test"]

results = {}
for temporal_type, spatial_type in product(config.MainConfig.TEMPORAL_TYPES, config.MainConfig.SPATIAL_TYPES):
    model = FlexibleCombinedModel(
        temporal_type=temporal_type,
        spatial_type=spatial_type,
        minmax_path=config.MainConfig.MINMAX_PATH,
        robust_path=config.MainConfig.ROBUST_PATH
    )
    metrics_sum = defaultdict(float)
    count = 0

    for (inputs_stage1, _), (_, targets_stage2) in tqdm(zip(stage_1, stage_2), 
                                                        total=len(stage_1),
                                                        desc=f"Evaluating {temporal_type}-{spatial_type}"):
        inputs = inputs_stage1.to(config.DEVICE)
        targets = targets_stage2.to(config.DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
        batch_metrics = evaluate_rainfall(outputs, targets)
        for k, v in batch_metrics.items():
            metrics_sum[k] += v
        count += 1

    metrics_avg = {k: v / count for k, v in metrics_sum.items()}

    results[f"{temporal_type}_{spatial_type}"] = metrics_avg

with open(config.MainConfig.JSON_PATH, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved all metrics to {config.MainConfig.JSON_PATH}")