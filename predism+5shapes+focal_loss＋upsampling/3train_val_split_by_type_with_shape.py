#oversample_focal_train_val_split.py
import os
import pandas as pd
import random

crop_metadata_csv = "../csv/cropped_building_metadata_with_shapes.csv"
output_dir = "../csv/type_train_val_split_with_shape_oversampled"
train_ratio = 0.8
random_seed = 42

# === upsampling factors ===
oversample_factors = {
    "earthquake": (1, 10, 20, 100),
    "flood": (1, 3, 3, 5),
    "hurricane": (1, 2, 2, 5),
    "tornado": (1, 3, 3, 5),
    "tsunami": (1, 5, 5, 5),
    "volcanic_eruption": (1, 5, 5, 5),
    "wildfire": (1, 5, 5, 5)
}

def split_crop_train_val_by_type_oversampled(crop_metadata_csv, output_dir, train_ratio=0.8, random_seed=42):
    df = pd.read_csv(crop_metadata_csv)
    os.makedirs(output_dir, exist_ok=True)

    disaster_types = sorted(df["disaster_type"].unique())
    print(f"✅ Found disaster types: {disaster_types}")

    for dtype in disaster_types:
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)

        df_c0 = df_type[df_type["label"] == 0]
        df_c1 = df_type[df_type["label"] == 1]
        df_c2 = df_type[df_type["label"] == 2]
        df_c3 = df_type[df_type["label"] == 3]

        # === oversample ===
        factor = oversample_factors.get(dtype, (1, 1, 1, 1))
        df_c0_ov = df_c0
        df_c1_ov = pd.concat([df_c1] * factor[1], ignore_index=True)
        df_c2_ov = pd.concat([df_c2] * factor[2], ignore_index=True)
        df_c3_ov = pd.concat([df_c3] * factor[3], ignore_index=True)

        # === integrating oversampled data ===
        df_balanced = pd.concat([df_c0_ov, df_c1_ov, df_c2_ov, df_c3_ov], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # === train/val split ===
        indices = list(range(len(df_balanced)))
        random.seed(random_seed)
        random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_df = df_balanced.iloc[train_indices].reset_index(drop=True)
        val_df = df_balanced.iloc[val_indices].reset_index(drop=True)

        dtype_name = df_type["disaster_type"].iloc[0]
        train_csv_path = os.path.join(output_dir, f"train_{dtype_name}.csv")
        val_csv_path = os.path.join(output_dir, f"val_{dtype_name}.csv")

        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        print(f"✅ {dtype_name}: Original {len(df_type)}, Oversampled {len(df_balanced)}, Train {len(train_df)}, Val {len(val_df)}")

if __name__ == "__main__":
    split_crop_train_val_by_type_oversampled(crop_metadata_csv, output_dir, train_ratio, random_seed)
