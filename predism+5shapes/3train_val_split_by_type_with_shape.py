# 2-4. data processing for resnet finetuning
import os
import pandas as pd
import random

crop_metadata_csv = "../csv/cropped_building_metadata_with_shapes.csv" 
output_dir = "../csv/type_train_val_split_with_shape" 
train_ratio = 0.8
random_seed = 42

def split_crop_train_val_by_type(crop_metadata_csv, output_dir, train_ratio=0.8, random_seed=42):
    df = pd.read_csv(crop_metadata_csv)
    
    os.makedirs(output_dir, exist_ok=True)

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

        dtype_str = df_type["disaster_name"].iloc[0]  
        dtype_name = df_type["disaster_type"].iloc[0]

        train_csv_path = os.path.join(output_dir, f"train_{dtype_name}.csv")
        val_csv_path = os.path.join(output_dir, f"val_{dtype_name}.csv")

        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        print(f"✅ Type {dtype_name}: Train {len(train_df)} buildings, Val {len(val_df)} buildings")

if __name__ == "__main__":
    split_crop_train_val_by_type(crop_metadata_csv, output_dir, train_ratio, random_seed)
