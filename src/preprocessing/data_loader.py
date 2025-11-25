import pandas as pd
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

from config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        print("Loading data files in parallel...")

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_mekong = executor.submit(self._load_mekong)
            future_sentinel_2 = executor.submit(self._load_sentinel_2)
            future_sentinel_1 = executor.submit(self._load_sentinel_1)
            future_target = executor.submit(self._load_target)

            df_mekong = future_mekong.result()
            df_sentinel_2 = future_sentinel_2.result()
            df_sentinel_1 = future_sentinel_1.result()
            df_target = future_target.result()

        return {
            'mekong': df_mekong,
            'sentinel_1': df_sentinel_1,
            'sentinel_2': df_sentinel_2,
            'target': df_target
        }

    def _load_mekong(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.mekong_csv_path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df

    def _load_target(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.target_csv_path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df

    def _load_sentinel_2(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.sentinel_2_csv_path)
        df['date'] = pd.to_datetime(df['image_date'], format='%Y-%m-%d_%H-%M-%S', errors='coerce')
        df.drop('image_date', axis=1, inplace=True)
        return df

    def _load_sentinel_1(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.sentinel_1_csv_path)
        df = df.drop(columns=['Total_Backscatter_mean', '_source_file'], errors='ignore')
        df['date'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df.drop('datetime', axis=1, inplace=True, errors='ignore')
        return df