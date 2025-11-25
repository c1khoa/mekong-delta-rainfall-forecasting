import pandas as pd
from typing import Dict

from config import Config
from data_loader import DataLoader
from data_splitter import DataSplitter
from data_processor import MekongDataProcessor
from tensor_builder import TensorBuilder
from visualizer import Visualizer


class MekongDataPipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.loader = DataLoader(self.config)
        self.splitter = DataSplitter(self.config)
        self.processor = MekongDataProcessor(self.config)
        self.tensor_builder = TensorBuilder(self.config)
        self.visualizer = Visualizer(self.config)

    def run(self) -> Dict:
        data = self.loader.load_all_data()

        splits = self.splitter.split_all(data)

        print("\nProcessing data...")
        processed = self._process_all_splits(splits)

        print("\nBuilding tensors...")
        tensors = self._build_all_tensors(processed)

        return {
            'data': data,
            'splits': splits,
            'processed': processed,
            'tensors': tensors
        }

    def _process_all_splits(self, splits: Dict) -> Dict:
        result = {}

        mekong_train, mekong_val, mekong_test = splits['mekong']
        mekong_train, mekong_val, mekong_test = self.processor.robust_scaling(
            mekong_train, mekong_val, mekong_test
        )
        mekong_train, mekong_val, mekong_test = self.processor.apply_pca_groups(
            mekong_train, mekong_val, mekong_test
        )
        result['mekong'] = (mekong_train, mekong_val, mekong_test)

        target_train, target_val, target_test = splits['target']
        result['target'] = self.processor.robust_scaling(
            target_train, target_val, target_test, target=True
        )

        s1_train, s1_val, s1_test = splits['sentinel_1']
        result['sentinel_1'] = self.processor.robust_scaling(s1_train, s1_val, s1_test)

        s2_train, s2_val, s2_test = splits['sentinel_2']
        result['sentinel_2'] = self.processor.robust_scaling(s2_train, s2_val, s2_test)

        return result

    def _build_all_tensors(self, processed: Dict) -> Dict:
        result = {}

        mekong_train, _, _ = processed['mekong']
        target_train, _, _ = processed['target']
        s1_train, _, _ = processed['sentinel_1']
        s2_train, _, _ = processed['sentinel_2']

        features = {
            'mekong': [c for c in mekong_train.columns if c not in self.config.exclude_cols],
            'target': [c for c in target_train.columns if c not in self.config.exclude_cols],
            'sentinel_1': [c for c in s1_train.columns if c not in self.config.exclude_cols],
            'sentinel_2': [c for c in s2_train.columns if c not in self.config.exclude_cols + ['AREA']]
        }

        print("Building Mekong tensors...")
        for split_name, split_data in [('train', 0), ('val', 1), ('test', 2)]:
            result[f'mekong_{split_name}'] = self.tensor_builder.build_mekong_tensor(
                processed['mekong'][split_data], features['mekong']
            )
            result[f'target_{split_name}'] = self.tensor_builder.build_mekong_tensor(
                processed['target'][split_data], features['target']
            )

        train_start = pd.Timestamp(self.config.train_start)
        val_start = pd.Timestamp(self.config.val_start)
        test_start = pd.Timestamp(self.config.test_start)
        train_end = pd.Timestamp(self.config.train_end)
        val_end = pd.Timestamp(self.config.val_end)
        test_end = pd.Timestamp(self.config.test_end)

        result['sentinel_1_train'] = self.tensor_builder.build_sentinel_tensor(
            processed['sentinel_1'][0], features['sentinel_1'],
            train_start.year, train_end.year, 'train'
        )
        result['sentinel_1_val'] = self.tensor_builder.build_sentinel_tensor(
            processed['sentinel_1'][1], features['sentinel_1'],
            val_start.year, val_end.year, 'val'
        )
        result['sentinel_1_test'] = self.tensor_builder.build_sentinel_tensor(
            processed['sentinel_1'][2], features['sentinel_1'],
            test_start.year, test_end.year, 'test'
        )

        result['sentinel_2_train'] = self.tensor_builder.build_sentinel_tensor(
            processed['sentinel_2'][0], features['sentinel_2'],
            train_start.year, train_end.year, 'train'
        )
        result['sentinel_2_val'] = self.tensor_builder.build_sentinel_tensor(
            processed['sentinel_2'][1], features['sentinel_2'],
            val_start.year, val_end.year, 'val'
        )
        result['sentinel_2_test'] = self.tensor_builder.build_sentinel_tensor(
            processed['sentinel_2'][2], features['sentinel_2'],
            test_start.year, test_end.year, 'test'
        )

        return result