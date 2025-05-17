# 2step_gat_damage_focal_test1.py
import os
import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.nn import GATConv

# === 入出力パス ===
graph_dir = "../graphs/2step_graphs_damage_test_only"
model_dir = "../models/focal_damage_gat_models_by_type"
output_dir = "../graphs/predicted_labels_step1"
os.makedirs(output_dir, exist_ok=True)

# === GATモデル定義 ===
class GATClassifier(nn.Module):
    def __init__(self, in_dim=521, hidden_dim=128, out_dim=4, heads=4, dropout=0.6):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return self.mlp(x)

# === 推論・保存 ===
def predict_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fname in os.listdir(graph_dir):
        if not fname.endswith("_step1.pt"):
            continue
        dtype = fname.replace("graph_", "").replace("_step1.pt", "")
        print(f"\n🧪 Predicting test graph for disaster type: {dtype}")

        graph_path = os.path.join(graph_dir, fname)
        model_path = os.path.join(model_dir, f"gat_{dtype}.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found for {dtype}, skipping.")
            continue

        g = torch.load(graph_path, weights_only=False)
        model = GATClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            y_pred = model(g.x.to(device), g.edge_index.to(device)).argmax(dim=1).cpu().numpy()

        pred_df = pd.DataFrame({"pred_label": y_pred})
        pred_df.to_csv(f"{output_dir}/pred_labels_{dtype}.csv", index=False)
        print(f"✅ Saved predictions for {dtype} → {output_dir}/pred_labels_{dtype}.csv")

# === 実行 ===
if __name__ == "__main__":
    predict_and_save()
