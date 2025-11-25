import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from typing import Tuple, List

from config import Config


class MekongDataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.robust_scaler = RobustScaler()
        self.robust_scaler_target = RobustScaler()

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        return [c for c in num_cols if c not in self.config.exclude_cols]

    def robust_scaling(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        target: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        num_cols = self._get_numeric_columns(df_train)

        if not num_cols:
            return df_train.copy(), df_val.copy(), df_test.copy()

        scaler = self.robust_scaler_target if target else self.robust_scaler

        df_train_scaled = df_train.copy()
        df_val_scaled = df_val.copy()
        df_test_scaled = df_test.copy()

        df_train_scaled[num_cols] = scaler.fit_transform(df_train[num_cols])
        df_val_scaled[num_cols] = scaler.transform(df_val[num_cols])
        df_test_scaled[num_cols] = scaler.transform(df_test[num_cols])

        return df_train_scaled, df_val_scaled, df_test_scaled

    def apply_pca(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        feature_list: List[str],
        new_col_prefix: str,
        n_components: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        available_features = [f for f in feature_list if f in df_train.columns]
        if not available_features:
            return df_train, df_val, df_test

        pca = PCA(n_components=n_components, random_state=42)

        train_feats = df_train[available_features].select_dtypes(include=[np.number])
        val_feats = df_val[available_features].select_dtypes(include=[np.number])
        test_feats = df_test[available_features].select_dtypes(include=[np.number])

        if train_feats.shape[1] == 0:
            return df_train, df_val, df_test

        pca_train = pca.fit_transform(train_feats)
        pca_val = pca.transform(val_feats)
        pca_test = pca.transform(test_feats)

        df_train_new = df_train.copy()
        df_val_new = df_val.copy()
        df_test_new = df_test.copy()

        for i in range(n_components):
            col_name = f"{new_col_prefix}_PC{i+1}"
            df_train_new[col_name] = pca_train[:, i]
            df_val_new[col_name] = pca_val[:, i]
            df_test_new[col_name] = pca_test[:, i]

        df_train_new.drop(columns=available_features, inplace=True, errors='ignore')
        df_val_new.drop(columns=available_features, inplace=True, errors='ignore')
        df_test_new.drop(columns=available_features, inplace=True, errors='ignore')

        return df_train_new, df_val_new, df_test_new

    def apply_pca_groups(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        n_components: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Applying PCA to feature groups...")
        for feature_list in self.config.pca_groups:
            prefix = "_".join(feature_list[:2])
            df_train, df_val, df_test = self.apply_pca(
                df_train, df_val, df_test, feature_list, prefix, n_components
            )
        return df_train, df_val, df_test