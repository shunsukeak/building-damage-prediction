# 2step_gat_damage_focal_testpre2
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from shapely import wkt

# === 入出力 ===
input_csv = "./cropped_test_building_metadata_with_shape.csv"
pred_dir = "./predicted_labels_step1"
# output_dir = "./2step_graphs_damage_test_only"
output_dir = "./2step_graphs_damage_test_only_label"
os.makedirs(output_dir, exist_ok=True)

k_neighbors = 8

def add_centroids(df):
    xs, ys = [], []
    for g in df["geometry"]:
        try:
            poly = wkt.loads(g)
            centroid = poly.centroid
            xs.append(centroid.x)
            ys.append(centroid.y)
        except:
            xs.append(0)
            ys.append(0)
    df["x"] = xs
    df["y"] = ys
    return df

df = pd.read_csv(input_csv)
df = add_centroids(df)
scaler = StandardScaler()
shape_cols = ["area", "perimeter", "aspect_ratio", "extent_ratio", "convexity"]
df[shape_cols] = scaler.fit_transform(df[shape_cols])

types = df["disaster_type"].unique()
print(f"📌 Disaster types (test): {types}")

# for dtype in types:
#     df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
#     if len(df_type) < 2:
#         print(f"⚠️ Skipping {dtype} (not enough nodes)")
#         continue

#     # === 予測ラベルの読み込み ===
#     pred_path = os.path.join(pred_dir, f"pred_labels_{dtype}.csv")
#     if not os.path.exists(pred_path):
#         print(f"⚠️ No prediction file for {dtype}, skipping")
#         continue
#     pred_labels = pd.read_csv(pred_path)["pred_label"].values

#     # === neighbor damage ratios 計算 ===
#     coords = df_type[["x", "y"]].values
#     knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(df_type)), algorithm='ball_tree')
#     knn.fit(coords)
#     neighbor_ratios = []
#     for i, neighbors in enumerate(knn.kneighbors(return_distance=False)):
#         neighbor_idxs = neighbors[1:]  # skip self
#         neighbor_preds = pred_labels[neighbor_idxs]
#         total_neighbors = len(neighbor_preds)
#         if total_neighbors == 0:
#             ratios = [0.0, 0.0, 0.0]
#         else:
#             minor_ratio = np.sum(neighbor_preds == 1) / total_neighbors
#             major_ratio = np.sum(neighbor_preds == 2) / total_neighbors
#             destroyed_ratio = np.sum(neighbor_preds == 3) / total_neighbors
#             ratios = [minor_ratio, major_ratio, destroyed_ratio]
#         neighbor_ratios.append(ratios)
#     neighbor_ratios = np.array(neighbor_ratios, dtype=np.float32)
for dtype in types:
    df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
    if len(df_type) < 2:
        print(f"⚠️ Skipping {dtype} (not enough nodes)")
        continue

    # === ground-truth ラベル使用 ===
    pred_labels = df_type["label"].values

    # === neighbor damage ratios 計算 ===
    coords = df_type[["x", "y"]].values
    knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(df_type)), algorithm='ball_tree')
    knn.fit(coords)
    neighbor_ratios = []
    for i, neighbors in enumerate(knn.kneighbors(return_distance=False)):
        neighbor_idxs = neighbors[1:]
        neighbor_preds = pred_labels[neighbor_idxs]
        total_neighbors = len(neighbor_preds)
        if total_neighbors == 0:
            ratios = [0.0, 0.0, 0.0]
        else:
            minor_ratio = np.sum(neighbor_preds == 1) / total_neighbors
            major_ratio = np.sum(neighbor_preds == 2) / total_neighbors
            destroyed_ratio = np.sum(neighbor_preds == 3) / total_neighbors
            ratios = [minor_ratio, major_ratio, destroyed_ratio]
        neighbor_ratios.append(ratios)
    neighbor_ratios = np.array(neighbor_ratios, dtype=np.float32)

    # === ノード特徴量（画像 + shape + hazard + neighbor rate） ===
    x_feats = []
    for idx, row in df_type.iterrows():
        img_vec = np.random.randn(512)  # ResNetダミー
        shape_vec = row[shape_cols].values.astype("float32")
        hazard = np.array([row["hazard_level"]], dtype="float32")
        neighbor = neighbor_ratios[idx]
        x_feats.append(np.concatenate([img_vec, shape_vec, hazard, neighbor]))

    x_tensor = torch.tensor(np.vstack(x_feats), dtype=torch.float)

    # === エッジ作成（k-NN） ===
    edge_index = []
    for i, neighbors in enumerate(knn.kneighbors(return_distance=False)):
        for j in neighbors[1:]:
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # === ラベル ===
    y = torch.tensor(df_type["label"].values, dtype=torch.long)

    # === graph 作成 ===
    data = Data(x=x_tensor, edge_index=edge_index, y=y)
    out_path = os.path.join(output_dir, f"graph_{dtype}_step2.pt")
    torch.save(data, out_path)
    print(f"✅ Saved step2 test graph: {out_path} ({len(df_type)} buildings)")
