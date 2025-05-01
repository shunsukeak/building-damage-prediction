import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# === 入出力パス ===
graph_dir = "./graphs_test_only"
model_dir = "./gcn_models_by_type"

# === GCNモデル定義 ===
class GCNClassifier(nn.Module):
    def __init__(self, in_dim=518, hidden_dim=128, out_dim=4):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        return self.mlp(x)

# === 評価関数 ===
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds, all_labels = [], []

    type_pred_dict = defaultdict(list)
    type_label_dict = defaultdict(list)

    for fname in os.listdir(graph_dir):
        if not fname.endswith("_test.pt"):
            continue
        dtype = fname.replace("graph_", "").replace("_test.pt", "")
        print(f"\n🧪 Evaluating test graph for disaster type: {dtype}")

        graph_path = os.path.join(graph_dir, fname)
        model_path = os.path.join(model_dir, f"gcn_{dtype}.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found for {dtype}, skipping.")
            continue

        g = torch.load(graph_path, weights_only=False)
        model = GCNClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        x = g.x.to(device)
        edge_index = g.edge_index.to(device)
        y_true = g.y.cpu().numpy()

        with torch.no_grad():
            out = model(x, edge_index)
            y_pred = out.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")

        all_preds.extend(y_pred)
        all_labels.extend(y_true)
        type_pred_dict[dtype].extend(y_pred)
        type_label_dict[dtype].extend(y_true)

    # === 全体評価 ===
    if len(all_preds) == 0:
        print("⚠️ No predictions made.")
        return

    print("\n📊 Overall GCN Test Evaluation (test-only graphs):")
    print("✅ Accuracy:", accuracy_score(all_labels, all_preds))
    print("✅ Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("✅ Classification Report:\n",
          classification_report(all_labels, all_preds, target_names=["no-damage", "minor", "major", "destroyed"]))

    # === 災害タイプ単位集計 ===
    print("\n🌍 Disaster-type summary:")
    disaster_accs = []
    for dtype in sorted(type_pred_dict.keys()):
        preds = np.array(type_pred_dict[dtype])
        labels = np.array(type_label_dict[dtype])
        correct = (preds == labels).sum()
        total = len(labels)
        disaster_acc = correct / total
        disaster_accs.append(disaster_acc)
        print(f"🌍 {dtype}: Accuracy = {disaster_acc:.4f} ({correct}/{total})")

    mean_disaster_acc = np.mean(disaster_accs)
    print(f"\n📊 Mean Disaster-type Accuracy: {mean_disaster_acc:.4f}")

# === 実行 ===
if __name__ == "__main__":
    evaluate()

# import os
# import torch
# import torch.nn as nn
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from tqdm import tqdm

# # === 入出力パス ===
# graph_dir = "./graphs_test_only"
# model_dir = "./gcn_models_by_type"

# # === GCNモデル定義 ===
# class GCNClassifier(nn.Module):
#     def __init__(self, in_dim=518, hidden_dim=128, out_dim=4):
#         super().__init__()
#         self.gcn1 = GCNConv(in_dim, hidden_dim)
#         self.gcn2 = GCNConv(hidden_dim, hidden_dim)
#         self.mlp = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, out_dim)
#         )

#     def forward(self, x, edge_index):
#         x = self.gcn1(x, edge_index)
#         x = self.gcn2(x, edge_index)
#         return self.mlp(x)

# # === 評価関数 ===
# def evaluate():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     all_preds, all_labels = [], []

#     for fname in os.listdir(graph_dir):
#         if not fname.endswith("_test.pt"): continue
#         dtype = fname.replace("graph_", "").replace("_test.pt", "")
#         print(f"\n🧪 Evaluating test graph for disaster type: {dtype}")

#         graph_path = os.path.join(graph_dir, fname)
#         model_path = os.path.join(model_dir, f"gcn_{dtype}.pt")
#         if not os.path.exists(model_path):
#             print(f"⚠️ Model not found for {dtype}, skipping.")
#             continue

#         # グラフとモデル読み込み
#         g = torch.load(graph_path, weights_only=False)
#         model = GCNClassifier().to(device)
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.eval()

#         x = g.x.to(device)
#         edge_index = g.edge_index.to(device)
#         y_true = g.y.cpu().numpy()

#         with torch.no_grad():
#             out = model(x, edge_index)
#             y_pred = out.argmax(dim=1).cpu().numpy()

#         acc = accuracy_score(y_true, y_pred)
#         print(f"✅ Accuracy: {acc:.4f}")
#         all_preds.extend(y_pred)
#         all_labels.extend(y_true)

#     # === 全体評価 ===
#     if len(all_preds) == 0:
#         print("⚠️ No predictions made.")
#     else:
#         print("\n📊 Overall GCN Test Evaluation (test-only graphs):")
#         print("✅ Accuracy:", accuracy_score(all_labels, all_preds))
#         print("✅ Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
#         print("✅ Classification Report:\n",
#               classification_report(all_labels, all_preds, target_names=["no-damage", "minor", "major", "destroyed"]))

# # === 実行 ===
# if __name__ == "__main__":
#     evaluate()

