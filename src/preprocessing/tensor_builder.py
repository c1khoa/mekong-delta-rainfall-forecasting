import numpy as np
import pandas as pd
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from config import Config
from numba_utils import fill_mekong_tensor_numba, compute_sentinel_slice


def process_sentinel_time_chunk(chunk_indices, time_array, dates_array,
                                x_starts, x_ends, y_starts, y_ends,
                                feat_vals, H, W, C):
    chunk_result = np.zeros((len(chunk_indices), H, W, C), dtype=np.float32)
    for i, t_idx in enumerate(chunk_indices):
        target_date = time_array[t_idx]
        chunk_result[i] = compute_sentinel_slice(
            dates_array, target_date, x_starts, x_ends,
            y_starts, y_ends, feat_vals, H, W, C
        )
    return chunk_result, chunk_indices


class TensorBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.lon_bins = np.arange(config.min_lon, config.max_lon, config.resolution)
        self.lat_bins = np.arange(config.min_lat, config.max_lat, config.resolution)
        self.H = len(self.lat_bins)
        self.W = len(self.lon_bins)

    def build_mekong_tensor(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        unique_dates = sorted(df['date'].unique())
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}

        T, C = len(unique_dates), len(features)

        feature_tensor = np.zeros((T, self.H, self.W, C), dtype=np.float32)
        count_tensor = np.zeros((T, self.H, self.W, C), dtype=np.int32)

        df['t_idx'] = df['date'].map(date_to_idx)
        df['x_start'] = np.clip(np.digitize(df['minx'].values, self.lon_bins) - 1, 0, self.W - 1)
        df['x_end'] = np.clip(np.digitize(df['maxx'].values, self.lon_bins), 0, self.W)
        df['y_start'] = np.clip(np.digitize(df['miny'].values, self.lat_bins) - 1, 0, self.H - 1)
        df['y_end'] = np.clip(np.digitize(df['maxy'].values, self.lat_bins), 0, self.H)

        valid_mask = (df['x_end'] > df['x_start']) & (df['y_end'] > df['y_start'])
        df = df[valid_mask]

        t_indices = df['t_idx'].values.astype(np.int32)
        x_starts = df['x_start'].values.astype(np.int32)
        x_ends = df['x_end'].values.astype(np.int32)
        y_starts = df['y_start'].values.astype(np.int32)
        y_ends = df['y_end'].values.astype(np.int32)
        feat_vals = df[features].values.astype(np.float32)

        feature_tensor, count_tensor = fill_mekong_tensor_numba(
            feature_tensor, count_tensor, t_indices,
            x_starts, x_ends, y_starts, y_ends, feat_vals
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            tensor = np.where(count_tensor > 0, feature_tensor / count_tensor, 0)

        return tensor

    def build_sentinel_tensor(
        self,
        df: pd.DataFrame,
        features: List[str],
        begin_year: int,
        end_year: int,
        subset: str = 'train'
    ) -> np.ndarray:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        C = len(features)
        time_index = self._get_time_index(begin_year, end_year, subset)
        T = len(time_index)

        df = self._precompute_spatial_indices(df, features)

        dates_array = (df['date'].values.astype('datetime64[D]') -
                      np.datetime64('1970-01-01')).astype(np.float32)
        time_array = (time_index.values.astype('datetime64[D]') -
                     np.datetime64('1970-01-01')).astype(np.float32)

        x_starts = df['x_start'].values.astype(np.int32)
        x_ends = df['x_end'].values.astype(np.int32)
        y_starts = df['y_start'].values.astype(np.int32)
        y_ends = df['y_end'].values.astype(np.int32)
        feat_vals = df[features].values.astype(np.float32)

        tensor = np.zeros((T, self.H, self.W, C), dtype=np.float32)

        if self.config.use_multiprocessing and T > 100:
            print(f"Building Sentinel tensor ({subset}) with multiprocessing...")

            chunks = [list(range(i, min(i + self.config.chunk_size, T)))
                     for i in range(0, T, self.config.chunk_size)]

            process_func = partial(
                process_sentinel_time_chunk,
                time_array=time_array,
                dates_array=dates_array,
                x_starts=x_starts,
                x_ends=x_ends,
                y_starts=y_starts,
                y_ends=y_ends,
                feat_vals=feat_vals,
                H=self.H,
                W=self.W,
                C=C
            )

            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = [executor.submit(process_func, chunk) for chunk in chunks]

                for future in tqdm(futures, desc=f"Processing {subset}"):
                    chunk_result, chunk_indices = future.result()
                    for i, t_idx in enumerate(chunk_indices):
                        tensor[t_idx] = chunk_result[i]
        else:
            for t_idx in tqdm(range(T), desc=f"Building Sentinel tensor ({subset})"):
                target_date = time_array[t_idx]
                tensor[t_idx] = compute_sentinel_slice(
                    dates_array, target_date, x_starts, x_ends,
                    y_starts, y_ends, feat_vals, self.H, self.W, C
                )

        return tensor

    def _precompute_spatial_indices(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        df = df.copy()

        df['x_start'] = np.clip(np.digitize(df['minx'].values, self.lon_bins) - 1, 0, self.W - 1)
        df['x_end'] = np.clip(np.digitize(df['maxx'].values, self.lon_bins), 0, self.W)
        df['y_start'] = np.clip(np.digitize(df['miny'].values, self.lat_bins) - 1, 0, self.H - 1)
        df['y_end'] = np.clip(np.digitize(df['maxy'].values, self.lat_bins), 0, self.H)

        valid_mask = (df['x_end'] > df['x_start']) & (df['y_end'] > df['y_start'])
        df = df[valid_mask].reset_index(drop=True)

        return df

    def _get_time_index(self, begin_year: int, end_year: int, subset: str) -> pd.DatetimeIndex:
        if subset == 'train':
            return pd.date_range(f"{begin_year}-01-01", f"{end_year}-12-31", freq='D')
        elif subset == 'val':
            return pd.date_range(f"{begin_year}-01-01", f"{end_year}-06-30", freq='D')
        else:
            return pd.date_range(f"{begin_year}-07-01", f"{end_year}-12-31", freq='D')