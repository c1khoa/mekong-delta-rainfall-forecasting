import numpy as np
import pandas as pd
from src import config
import os
from datetime import datetime, date, timedelta
from src.dataset import FolderTimeSeriesDataset
from src.pipeline import FlexibleCombinedModel
import torch
import json
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


province_bbx = pd.read_csv(config.MainConfig.BBX_PATH)

stage1_models = ["ConvLSTM", "ConvGRU", "TrajGRU"]
stage2_models = ["UNet", "FPN", "UNet3Plus"]

def rainfall_by_province(rain, province_bbx):
    results = []

    for _, row in province_bbx.iterrows():
        values = []

        for i in range(config.H):
            for j in range(config.W):
                lat = config.lats[i]
                lon = config.lons[j]

                if (
                    row.minx <= lon <= row.maxx
                    and row.miny <= lat <= row.maxy
                ):
                    values.append(rain[i, j])

        results.append({
            "province": row.province,
            "num_pixels": len(values),
            "avg_rainfall": np.mean(values) if values else np.nan
        })

    return pd.DataFrame(results)

def main_predict(model_stage1, model_stage2, day, month, year):
    test_path = os.path.join(config.Dataset.PATH, "test")
    test_dataset = FolderTimeSeriesDataset(test_path)[-1][0]

    start_date = datetime(2024, 12, 31)
    end_date = datetime(year, month, day)
    n_days = (end_date - start_date).days

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FlexibleCombinedModel(
        model_stage1,
        model_stage2,
        config.MINMAX_PATH,
        config.ROBUST_PATH
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(test_dataset, n_days=n_days)

    return outputs.squeeze(0).squeeze(0).cpu().numpy()  # (35, 35)

def predict_daily_images_for_year(model_stage1, model_stage2, year):
    daily_results = []

    cur = date(year, 1, 1)
    end = date(year, 12, 31)

    total_days = (end - cur).days + 1

    for _ in tqdm(
        range(total_days),
        desc=f"{model_stage1}+{model_stage2} | {year}",
        leave=False
    ):
        rain_img = main_predict(
            model_stage1,
            model_stage2,
            cur.day,
            cur.month,
            cur.year
        )

        daily_results.append({
            "date": cur,
            "year": cur.year,
            "month": cur.month,
            "rain": rain_img
        })

        cur += timedelta(days=1)

    return daily_results


def daily_province_rainfall(daily_images, province_bbx):
    records = []

    for item in daily_images:
        df = rainfall_by_province(item["rain"], province_bbx)
        df["date"] = item["date"]
        df["year"] = item["year"]
        df["month"] = item["month"]
        records.append(df)

    return pd.concat(records, ignore_index=True)

def monthly_province_average(df_daily):
    return (
        df_daily
        .groupby(["year", "month", "province"])
        .agg(
            avg_monthly_rainfall=("avg_rainfall", "mean"),
            avg_num_pixels=("num_pixels", "mean")
        )
        .reset_index()
    )

def run_monthly_rainfall_pipeline(model_stage1, model_stage2):
    all_daily = []

    for year in [2025, 2026, 2027]:
        print(f"Processing year {year} ...")

        daily_imgs = predict_daily_images_for_year(
            model_stage1,
            model_stage2,
            year
        )

        df_daily = daily_province_rainfall(
            daily_imgs,
            province_bbx
        )

        all_daily.append(df_daily)

    df_daily_all = pd.concat(all_daily, ignore_index=True)
    df_monthly = monthly_province_average(df_daily_all)

    return df_monthly

stage1_models = ["ConvLSTM", "ConvGRU", "TrajGRU"]
stage2_models = ["UNet", "FPN", "UNet3Plus"]

output_dir = os.path.abspath("output")
os.makedirs(output_dir, exist_ok=True)

all_results = []

for stage1_name, stage2_name in tqdm(
    list(product(stage1_models, stage2_models)),
    desc="Model combinations"
):
    print(f"\n🚀 Running: {stage1_name} + {stage2_name}")

    df_monthly = run_monthly_rainfall_pipeline(
        stage1_name,
        stage2_name
    )

    df_monthly["stage1_model"] = stage1_name
    df_monthly["stage2_model"] = stage2_name

    results = df_monthly.to_dict(orient="records")

    filename = f"monthly_rainfall_{stage1_name}_{stage2_name}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved: {output_path}")
