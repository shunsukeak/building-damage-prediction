# upsample_gat_train2.py
import os
import pandas as pd
import random
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# === 入出力 ===
input_csv = "./all_buildings_for_graph.csv"
output_dir = "./upsample_graphs_by_type"
os.makedirs(output_dir, exist_ok=True)

# === oversample 倍率設定 ===
oversample_factors = {
    "earthquake": (1, 10, 20, 100),
    "flood": (1, 3, 3, 5),
    "hurricane": (1, 2, 2, 5),
    "tornado": (1, 3, 3, 5),
    "tsunami": (1, 5, 5, 5),
    "volcanic_eruption": (1, 5, 5, 5),
    "wildfire": (1, 5, 5, 5)
}

# === oversample 実行関数 ===
def oversample_data(df):
    oversampled_dfs = []
    disaster_types = sorted(df["disaster_type"].unique())
    print(f"✅ Found disaster types: {disaster_types}")

    for dtype in disaster_types:
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
        df_c0 = df_type[df_type["label"] == 0]
        df_c1 = df_type[df_type["label"] == 1]
        df_c2 = df_type[df_type["label"] == 2]
        df_c3 = df_type[df_type["label"] == 3]

        factor = oversample_factors.get(dtype, (1, 1, 1, 1))
        df_c0_ov = df_c0
        df_c1_ov = pd.concat([df_c1] * factor[1], ignore_index=True)
        df_c2_ov = pd.concat([df_c2] * factor[2], ignore_index=True)
        df_c3_ov = pd.concat([df_c3] * factor[3], ignore_index=True)

        df_balanced = pd.concat([df_c0_ov, df_c1_ov, df_c2_ov, df_c3_ov], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        oversampled_dfs.append(df_balanced)

        print(f"✅ {dtype}: Original {len(df_type)}, Oversampled {len(df_balanced)}")

    df_all = pd.concat(oversampled_dfs, ignore_index=True)
    return df_all

# === グラフ作成関数 ===
def build_graphs(df, output_dir, k=8):
    disaster_types = sorted(df["disaster_type"].unique())

    for dtype in disaster_types:
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
        if len(df_type) < 5:
            print(f"⚠️ Skipping {dtype} (too few nodes: {len(df_type)})")
            continue

        x_feats = df_type[["area", "perimeter", "aspect_ratio", "extent_ratio", "convexity", "hazard_level"]].values
        x_feats = torch.tensor(x_feats, dtype=torch.float)
        y = torch.tensor(df_type["label"].values, dtype=torch.long)
        coords = df_type[["x", "y"]].values

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        edge_index = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):
                src = i
                dst = indices[i][j]
                edge_index.append([src, dst])
                edge_index.append([dst, src])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data = Data(x=x_feats, edge_index=edge_index, y=y)
        data.disaster_type = dtype

        save_path = os.path.join(output_dir, f"graph_{dtype}.pt")
        torch.save(data, save_path)
        print(f"💾 Saved graph: {save_path} ({x_feats.shape[0]} nodes)")

# === 実行 ===
if __name__ == "__main__":
    df_input = pd.read_csv(input_csv)
    df_oversampled = oversample_data(df_input)
    build_graphs(df_oversampled, output_dir, k=8)
