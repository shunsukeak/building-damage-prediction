# gat_damage_focal_add.py
import os
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# === 入出力 ===
input_csv = "../csv/all_buildings_for_graph.csv"
output_dir = "../graphs/graphs_by_type_damage"
os.makedirs(output_dir, exist_ok=True)

# === グラフ作成 ===
def build_graphs(csv_path, output_dir, k=8):
    df = pd.read_csv(csv_path)
    disaster_types = sorted(df["disaster_type"].unique())
    print(f"✅ Found disaster types: {disaster_types}")

    for dtype in disaster_types:
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
        if len(df_type) < 5:
            print(f"⚠️ Skipping {dtype} (too few nodes: {len(df_type)})")
            continue

        # === ノード特徴 ===
        x_feats = df_type[["area", "perimeter", "aspect_ratio", "extent_ratio", "convexity", "hazard_level"]].values
        x_feats = torch.tensor(x_feats, dtype=torch.float)

        # === ラベル ===
        y = torch.tensor(df_type["label"].values, dtype=torch.long)

        # === 位置情報から近傍計算 ===
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

        # === neighbor damage ratio 計算 ===
        neighbor_ratios = []
        for i in range(len(df_type)):
            neighbor_idxs = indices[i][1:]  # 自分を除く
            neighbor_labels = y[neighbor_idxs]
            total_neighbors = len(neighbor_labels)
            if total_neighbors == 0:
                ratios = [0.0, 0.0, 0.0]
            else:
                minor_ratio = (neighbor_labels == 1).sum().item() / total_neighbors
                major_ratio = (neighbor_labels == 2).sum().item() / total_neighbors
                destroyed_ratio = (neighbor_labels == 3).sum().item() / total_neighbors
                ratios = [minor_ratio, major_ratio, destroyed_ratio]
            neighbor_ratios.append(ratios)

        neighbor_ratios = torch.tensor(neighbor_ratios, dtype=torch.float)

        # === ノード特徴結合 ===
        x_final = torch.cat([x_feats, neighbor_ratios], dim=1)  # (N, 9)

        # === Data オブジェクト作成 ===
        data = Data(x=x_final, edge_index=edge_index, y=y)
        data.disaster_type = dtype

        save_path = os.path.join(output_dir, f"graph_{dtype}.pt")
        torch.save(data, save_path)
        print(f"💾 Saved graph: {save_path} ({x_final.shape[0]} nodes)")

# === 実行 ===
if __name__ == "__main__":
    build_graphs(input_csv, output_dir, k=8)
