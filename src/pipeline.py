from src.dataset import FolderTimeSeriesDataset, TensorTimeSeriesDataset
from src.models.convlstm import ConvLSTMForecaster
from src.models.unet import UNet
from src.models.convgru import ConvGRUForecaster
from src.models.trajgru import TrajGRU
from src.models.fpn import RainfallFPN
from src.models.unet3plus import UNet3Plus
import joblib, geopandas as gpd, torch, torch.nn as nn
from datetime import datetime
from src import config
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def main_predict(model_stage1, model_stage2, day, month, year):
    test_path = os.path.join(config.Dataset.PATH, "test")
    test_dataset = FolderTimeSeriesDataset(test_path)[-1][0]
    start_date = datetime(2024, 12, 31)
    end_date = datetime(year, month, day)
    n_days = (end_date - start_date).days
    print(f"⏳ Số ngày cần dự đoán từ 31/12/2024: {n_days} ngày")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FlexibleCombinedModel(model_stage1, model_stage2, config.MINMAX_PATH, config.ROBUST_PATH)
    model = model.to(device)
    model.eval()

    outputs = model(test_dataset, n_days=n_days)
    
    outputs = outputs.squeeze(0).squeeze(0)
    return torch.clamp(outputs, min=0.0)


class FlexibleCombinedModel(nn.Module):
    def __init__(self, temporal_type, spatial_type, minmax_path, robust_path):
        super().__init__()
        self.temporal_type = temporal_type
        self.spatial_type = spatial_type
        
        self.temporal_model = self._build_temporal(temporal_type)
        self.spatial_model = self._build_spatial(spatial_type)
        
        self.minmax_scaler = joblib.load(minmax_path)
        self.robust_scaler = joblib.load(robust_path)
        
    
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
        
    def autoregressive_forecast_last(self, initial_seq, n_days=7):
        self.temporal_model.eval()
        initial_seq = initial_seq.unsqueeze(0)
        current_seq = initial_seq.to(device)
        preds = []

        with torch.no_grad():
            for _ in range(n_days):
                out = self.temporal_model(current_seq)
                preds.append(out[0, 0].cpu().numpy())
                current_seq = torch.cat([current_seq[:, 1:], out], dim=1)

        return torch.from_numpy(np.array(preds)[-1]).unsqueeze(0).float().to(device)
    
    def forward(self, x, n_days=7):
        feat = self.autoregressive_forecast_last(x, n_days) # [B, time_out=1, C, H, W]
        feat = feat.squeeze(1) # [B, C, H, W]
        feat = self.spatial_model(feat) # [B, 1, H, W]
        
        feat = feat.detach().cpu().numpy()
        N, C, H, W = feat.shape
        feat = feat.reshape(-1, 1)
        feat = self.robust_scaler.inverse_transform(self.minmax_scaler.inverse_transform(feat)).reshape(N, C, H, W)
        
        return torch.from_numpy(feat).to(config.DEVICE)