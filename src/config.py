import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GADM_PATH = "models/gadm41_VNM_1.json"
MINMAX_PATH = "models/minmax_scaler_target_train.pkl"
ROBUST_PATH = "models/robust_scaler_target_train.pkl"
class Dataset:
    PATH = "data/preprocessed/"
    TIME_IN = 30
    TIME_OUT = 1
    BATCH_SIZE_1 = 8
    BATCH_SIZE_2 = 8
    NUM_WORKERS = 0
    PIN_MEMORY = True

class ConvLSTM:
    PATH = "models/convlstm.pth"
    INPUT_CHANNELS = 24
    HIDDEN_DIMS = [64, 32, 16]                  
    KERNEL_SIZE = 3

class ConvGRU:
    PATH = "models/convgru.pth"
    INPUT_CHANNELS = 24
    HIDDEN_DIMS = [64, 32, 16]
    KERNEL_SIZE = 3
    
class TrajGRU:
    PATH = "models/trajgru.pth"
    TIME_OUT = 1
    INPUT_CHANNELS = 24
    HIDDEN_DIMS = [16, 16, 8]
    OUTPUT_CHANNELS = 24
    L = 5
    
class Unet:
    PATH = "models/unet.pth"
    INPUT_CHANNELS = 24
    OUT_CHANNELS = 1
    FEATURES = [32, 64, 128, 256]
    
class FPN:
    PATH = "models/fpn.pth"
    INPUT_CHANNELS = 24
    BASE_CHANNELS = 32
    FPN_CHANNELS = 64
    
class UNet3Plus:
    PATH = "models/unet3plus.pth"
    INPUT_CHANNELS = 24
    OUT_CHANNELS = 1
    BASE_CHANNELS = 8
    
class MainConfig:
    MINMAX_PATH = "data/preprocessed/minmax_scaler_target.pkl"
    ROBUST_PATH = "data/preprocessed/robust_scaler_target.pkl"
    TEMPORAL_TYPES = ["ConvLSTM", "ConvGRU", "TrajGRU"]
    SPATIAL_TYPES = ["UNet", "FPN", "UNet3Plus"]
    JSON_PATH = "results.json"
    BBX_PATH = "data/raw/mekong_provinces_bbox.csv"

H, W = 35, 35

lon_min, lon_max = 103.5, 107.0
lat_min, lat_max = 8.0, 11.5

dlon = (lon_max - lon_min) / W
dlat = (lat_max - lat_min) / H

lons = lon_min + (np.arange(W) + 0.5) * dlon
lats = lat_max - (np.arange(H) + 0.5) * dlat

    