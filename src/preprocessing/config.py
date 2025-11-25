import multiprocessing as mp
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    mekong_csv_path: str = "../../data/raw/mekong_data_final_2019_2024.csv"
    sentinel_2_csv_path: str = "../../data/raw/sentinel2_extract_data.csv"
    sentinel_1_csv_path: str = "../../data/raw/sentinel1_extract_data.csv"
    target_csv_path: str = "../../data/raw/target_precipitation.csv"
    vn_adm1_path: str = '../../data/raw/gadm41_VNM_1.json'
    
    output_dir: str = "../output"
    dataset_ts_path: str = None
    robust_scaler_path: str = None
    minmax_scaler_path: str = None

    train_start: str = "2019-01-01"
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    test_end: str = "2024-12-31"

    min_lon: float = 103.5
    max_lon: float = 107.0
    min_lat: float = 8.0
    max_lat: float = 11.5
    resolution: float = 0.1

    n_workers: int = None
    chunk_size: int = 50
    use_multiprocessing: bool = True

    exclude_cols: List[str] = None

    pca_groups: List[List[str]] = None

    mekong_provinces: List[str] = None

    def __post_init__(self):
        if self.n_workers is None:
            self.n_workers = max(1, mp.cpu_count() - 1)

        if self.exclude_cols is None:
            self.exclude_cols = ['minx', 'maxx', 'miny', 'maxy', 'province', 'date', 'geometry', 'id']

        if self.pca_groups is None:
            self.pca_groups = [
                ['Rainf_tavg', 'Rainf_f_tavg', 'CanopInt_inst', 'ECanop_tavg', 'Lwnet_tavg'],
                ['SoilMoi0_10cm_inst', 'Qair_f_inst', 'SoilMoi10_40cm_inst', 'volumetric_soil_water_layer_3', 'dewpoint_temperature_2m']
            ]

        if self.mekong_provinces is None:
            self.mekong_provinces = [
                "LongAn", "TiềnGiang", "BếnTre", "TràVinh", "VĩnhLong",
                "ĐồngTháp", "AnGiang", "CầnThơ", "HậuGiang",
                "SócTrăng", "BạcLiêu", "CàMau", "KiênGiang"
            ]
        
        if self.dataset_ts_path is None:
            import os
            self.dataset_ts_path = self.output_dir
        
        if self.robust_scaler_path is None:
            import os
            self.robust_scaler_path = os.path.join(self.output_dir, "robust_scaler_target.pkl")
        
        if self.minmax_scaler_path is None:
            import os
            self.minmax_scaler_path = os.path.join(self.output_dir, "minmax_scaler_target.pkl")