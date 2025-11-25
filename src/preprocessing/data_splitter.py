import pandas as pd
from typing import Tuple, Dict

from config import Config


class DataSplitter:
    def __init__(self, config: Config):
        self.config = config
        self.train_start = pd.Timestamp(config.train_start)
        self.train_end = pd.Timestamp(config.train_end)
        self.val_start = pd.Timestamp(config.val_start)
        self.val_end = pd.Timestamp(config.val_end)
        self.test_start = pd.Timestamp(config.test_start)
        self.test_end = pd.Timestamp(config.test_end)

    def split(self, df: pd.DataFrame, date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        date_series = df[date_col]

        train_mask = (date_series >= self.train_start) & (date_series <= self.train_end)
        val_mask = (date_series >= self.val_start) & (date_series <= self.val_end)
        test_mask = (date_series >= self.test_start) & (date_series <= self.test_end)

        return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()

    def split_all(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        print("Splitting data into train/val/test...")
        result = {}
        for name, df in data_dict.items():
            train, val, test = self.split(df)
            result[name] = (train, val, test)
            print(f"{name:12s} - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        return result