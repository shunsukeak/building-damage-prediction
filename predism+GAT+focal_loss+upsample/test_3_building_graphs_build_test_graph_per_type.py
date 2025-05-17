import os
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from shapely import wkt

input_csv = "../csv/cropped_test_building_metadata_with_shape.csv"
output_dir = "../graphs/graphs_test_only"
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

for dtype in types:
    df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
    if len(df_type) < 2:
        print(f"⚠️ Skipping {dtype} (not enough nodes)")
        continue

    x_feats = []
    for _, row in df_type.iterrows():
        # dummy image vector (512 dim), normally from ResNet
        img_vec = np.random.randn(512)
        shape_vec = row[shape_cols].values.astype("float32")
        hazard = np.array([row["hazard_level"]], dtype="float32")
        x_feats.append(np.concatenate([img_vec, shape_vec, hazard]))

    x_tensor = torch.tensor(np.vstack(x_feats), dtype=torch.float)

    coords = df_type[["x", "y"]].values
    knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(df_type)), algorithm='ball_tree')
    knn.fit(coords)
    edge_index = []
    for i, neighbors in enumerate(knn.kneighbors(return_distance=False)):
        for j in neighbors[1:]:  # skip self
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    y = torch.tensor(df_type["label"].values, dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y)

    out_path = os.path.join(output_dir, f"graph_{dtype}_test.pt")
    torch.save(data, out_path)
    print(f"✅ Saved test graph: {out_path} ({len(df_type)} buildings)")
