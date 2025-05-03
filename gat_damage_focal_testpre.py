# gat_damage_focal_testpre.py
import os
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

# === 入出力 ===
input_csv = "./cropped_test_building_metadata_with_shape.csv"
output_dir = "./graphs_damage_test_only"
os.makedirs(output_dir, exist_ok=True)

def build_test_graphs(csv_path, output_dir, k=8):
    df = pd.read_csv(csv_path)
    disaster_types = sorted(df["disaster_type"].unique())
    print(f"✅ Found disaster types: {disaster_types}")

    for dtype in disaster_types:
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
        if len(df_type) < 5:
            print(f"⚠️ Skipping {dtype} (too few nodes: {len(df_type)})")
            continue

        # === ノード特徴（shape系のみ） ===
        x_feats = df_type[["area", "perimeter", "aspect_ratio", "extent_ratio", "convexity", "hazard_level"]].values
        x_feats = torch.tensor(x_feats, dtype=torch.float)

        # === ラベル（これは保存用だけ、計算には使わない） ===
        y = torch.tensor(df_type["label"].values, dtype=torch.long)

        # === 近傍グラフ構築（x, y 座標ベース） ===
        coords = df_type[["x", "y"]].values
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        edge_index = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # 自己ループ除外
                src = i
                dst = indices[i][j]
                edge_index.append([src, dst])
                edge_index.append([dst, src])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # === Dataオブジェクト作成 ===
        data = Data(x=x_feats, edge_index=edge_index, y=y)
        data.disaster_type = dtype

        save_path = os.path.join(output_dir, f"graph_{dtype}_test.pt")
        torch.save(data, save_path)
        print(f"💾 Saved test graph: {save_path} ({x_feats.shape[0]} nodes)")

if __name__ == "__main__":
    build_test_graphs(input_csv, output_dir, k=8)
