# 3_GAT_input_prepare.py

import os
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

input_csv = "../stage1_damage_only.csv"      # Stage1 damage CSV
output_dir = "../graphs_damage_only_by_type" 
os.makedirs(output_dir, exist_ok=True)

def build_damage_graphs(csv_path, output_dir, k=8):
    df = pd.read_csv(csv_path)

    disaster_types = sorted(df["disaster_type"].unique())
    print(f"✅ Found disaster types: {disaster_types}")

    for dtype in disaster_types:
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
        if len(df_type) < 5:
            print(f"⚠️ Skipping {dtype} (too few damage nodes: {len(df_type)})")
            continue

        # shape + hazard
        x_feats = df_type[["area", "perimeter", "aspect_ratio", "extent_ratio", "convexity", "hazard_level"]].values
        x_feats = torch.tensor(x_feats, dtype=torch.float)

        y = torch.tensor(df_type["label"].values, dtype=torch.long)

        #（k-NN）
        coords = df_type[["x", "y"]].values
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(df_type)), algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        edge_index = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # skip self
                src = i
                dst = indices[i][j]
                edge_index.append([src, dst])
                edge_index.append([dst, src])  

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # PyG Data 
        data = Data(x=x_feats, edge_index=edge_index, y=y)
        data.disaster_type = dtype

        save_path = os.path.join(output_dir, f"graph_{dtype}_damage.pt")
        torch.save(data, save_path)
        print(f"💾 Saved damage graph: {save_path} ({x_feats.shape[0]} nodes)")


if __name__ == "__main__":
    build_damage_graphs(input_csv, output_dir, k=8)
