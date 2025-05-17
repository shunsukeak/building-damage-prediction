import os
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

input_csv = "../csv/all_buildings_for_graph.csv"
output_dir = "../graphs/graphs_by_type"
os.makedirs(output_dir, exist_ok=True)

# === graphs ===
def build_graphs(csv_path, output_dir, k=8):
    df = pd.read_csv(csv_path)

    disaster_types = sorted(df["disaster_type"].unique())
    print(f"✅ Found disaster types: {disaster_types}")

    for dtype in disaster_types:
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
        if len(df_type) < 5:
            print(f"⚠️ Skipping {dtype} (too few nodes: {len(df_type)})")
            continue

        # === node features ===
        x_feats = df_type[["area", "perimeter", "aspect_ratio", "extent_ratio", "convexity", "hazard_level"]].values
        x_feats = torch.tensor(x_feats, dtype=torch.float)

        # === label ===
        y = torch.tensor(df_type["label"].values, dtype=torch.long)

        # === neighbor graph ===
        coords = df_type[["x", "y"]].values
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        edge_index = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # remove self-loop
                src = i
                dst = indices[i][j]
                edge_index.append([src, dst])
                edge_index.append([dst, src])  # undirected graph
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # === PyG Data object ===
        data = Data(x=x_feats, edge_index=edge_index, y=y)
        data.disaster_type = dtype

        save_path = os.path.join(output_dir, f"graph_{dtype}.pt")
        torch.save(data, save_path)
        print(f"💾 Saved graph: {save_path} ({x_feats.shape[0]} nodes)")

if __name__ == "__main__":
    build_graphs(input_csv, output_dir, k=8)
