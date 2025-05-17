# gat_focal_test.py
import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd

# === 入出力パス ===
graph_dir = "../graphs/graphs_test_only"
model_dir = "../models/focal_gat_models_by_type"  
metadata_csv = "../csv/cropped_test_building_metadata_with_shape.csv"

# === GATモデル定義 ===
class GATClassifier(nn.Module):
    def __init__(self, in_dim=518, hidden_dim=128, out_dim=4, heads=4, dropout=0.6):
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

# === 評価関数 ===
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds, all_labels = [], []

    type_pred_dict = defaultdict(list)
    type_label_dict = defaultdict(list)
    disaster_name_pred_dict = defaultdict(list)
    disaster_name_label_dict = defaultdict(list)

    # metadata 読み込み
    df_meta = pd.read_csv(metadata_csv)
    img_paths = df_meta["image_path"].tolist()
    disaster_names = [os.path.basename(path).split("_")[0] for path in img_paths]

    idx = 0

    for fname in os.listdir(graph_dir):
        if not fname.endswith("_test.pt"):
            continue
        dtype = fname.replace("graph_", "").replace("_test.pt", "")
        print(f"\n🧪 Evaluating test graph for disaster type: {dtype}")

        graph_path = os.path.join(graph_dir, fname)
        model_path = os.path.join(model_dir, f"gat_{dtype}.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found for {dtype}, skipping.")
            continue

        g = torch.load(graph_path, weights_only=False)
        model = GATClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        x = g.x.to(device)
        edge_index = g.edge_index.to(device)
        y_true = g.y.cpu().numpy()

        with torch.no_grad():
            y_pred = model(x, edge_index).argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")

        all_preds.extend(y_pred)
        all_labels.extend(y_true)
        type_pred_dict[dtype].extend(y_pred)
        type_label_dict[dtype].extend(y_true)

        # disaster 名ごとの保存
        num_nodes = len(y_true)
        for i in range(num_nodes):
            disaster_name = disaster_names[idx]
            disaster_name_pred_dict[disaster_name].append(y_pred[i])
            disaster_name_label_dict[disaster_name].append(y_true[i])
            idx += 1

    # === 全体評価 ===
    if len(all_preds) == 0:
        print("⚠️ No predictions made.")
        return

    print("\n📊 Overall GAT Test Evaluation (test-only graphs):")
    print("✅ Accuracy:", accuracy_score(all_labels, all_preds))
    print("✅ Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("✅ Classification Report:\n",
          classification_report(all_labels, all_preds, target_names=["no-damage", "minor", "major", "destroyed"]))

    # === 災害タイプ単位 ===
    print("\n🌍 Disaster-type summary:")
    disaster_type_accs = []
    for dtype in sorted(type_pred_dict.keys()):
        preds = np.array(type_pred_dict[dtype])
        labels = np.array(type_label_dict[dtype])
        correct = (preds == labels).sum()
        total = len(labels)
        acc = correct / total
        disaster_type_accs.append(acc)
        print(f"🌍 {dtype}: Accuracy = {acc:.4f} ({correct}/{total})")
    mean_type_acc = np.mean(disaster_type_accs)
    print(f"\n📊 Mean Disaster-type Accuracy: {mean_type_acc:.4f}")

    # === 災害名単位 ===
    print("\n🌋 Disaster-name summary:")
    disaster_name_accs = []
    for dname in sorted(disaster_name_pred_dict.keys()):
        preds = np.array(disaster_name_pred_dict[dname])
        labels = np.array(disaster_name_label_dict[dname])
        correct = (preds == labels).sum()
        total = len(labels)
        acc = correct / total
        disaster_name_accs.append(acc)
        print(f"🌋 {dname}: Accuracy = {acc:.4f} ({correct}/{total})")
    mean_name_acc = np.mean(disaster_name_accs)
    print(f"\n📊 Mean Disaster-name Accuracy: {mean_name_acc:.4f}")

# === 実行 ===
if __name__ == "__main__":
    evaluate()
