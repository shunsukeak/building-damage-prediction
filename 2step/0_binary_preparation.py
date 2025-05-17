# prepare_binary_split.py

import os
import pandas as pd
import random

# === パラメータ設定 ===
input_csv = "../cropped_building_metadata_with_shapes.csv"
output_dir = "../type_train_val_split_with_shape_binary"
train_ratio = 0.8
random_seed = 42

# === ディレクトリ作成 ===
os.makedirs(output_dir, exist_ok=True)

# === main 処理 ===
def prepare_binary_labels_and_split(input_csv, output_dir, train_ratio=0.8, random_seed=42):
    df = pd.read_csv(input_csv)

    # === binary_label 列を作成: 0 → no-damage, 1 → damaged (minor/major/destroyed)
    df["binary_label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

    # === disaster_typeごとにsplit
    disaster_types = sorted(df["disaster_type"].unique())
    print(f"✅ Found disaster types: {disaster_types}")

    for dtype in disaster_types:
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)

        indices = list(range(len(df_type)))
        random.seed(random_seed)
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_df = df_type.iloc[train_indices].reset_index(drop=True)
        val_df = df_type.iloc[val_indices].reset_index(drop=True)

        train_csv_path = os.path.join(output_dir, f"train_{dtype}.csv")
        val_csv_path = os.path.join(output_dir, f"val_{dtype}.csv")

        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        print(f"✅ {dtype}: Train {len(train_df)}, Val {len(val_df)}")

# === 実行 ===
if __name__ == "__main__":
    prepare_binary_labels_and_split(input_csv, output_dir, train_ratio, random_seed)
