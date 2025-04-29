import os
import pandas as pd
import random

# === パラメータ設定 ===
crop_metadata_csv = "./cropped_building_metadata.csv"  # crop画像のmetadata
output_dir = "./crop_train_val_split"
train_ratio = 0.8
random_seed = 42

# === 処理開始 ===
def split_crop_train_val(crop_metadata_csv, output_dir, train_ratio=0.8, random_seed=42):
    df = pd.read_csv(crop_metadata_csv)

    os.makedirs(output_dir, exist_ok=True)

    disasters = sorted(df["disaster_name"].unique())
    print(f"✅ Found disasters: {disasters}")

    for disaster in disasters:
        df_disaster = df[df["disaster_name"] == disaster].reset_index(drop=True)

        building_indices = list(range(len(df_disaster)))
        random.seed(random_seed)
        random.shuffle(building_indices)

        split_idx = int(len(building_indices) * train_ratio)
        train_indices = building_indices[:split_idx]
        val_indices = building_indices[split_idx:]

        train_df = df_disaster.iloc[train_indices].reset_index(drop=True)
        val_df = df_disaster.iloc[val_indices].reset_index(drop=True)

        # 保存
        train_csv_path = os.path.join(output_dir, f"train_{disaster}.csv")
        val_csv_path = os.path.join(output_dir, f"val_{disaster}.csv")

        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        print(f"✅ Disaster {disaster}: Train {len(train_df)} buildings, Val {len(val_df)} buildings")

# === 実行例 ===
if __name__ == "__main__":
    split_crop_train_val(crop_metadata_csv, output_dir, train_ratio, random_seed)
