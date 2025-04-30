import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

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

    for fname in os.listdir(graph_dir):
        if not fname.endswith("_test.pt"): continue
        dtype = fname.replace("graph_", "").replace("_test.pt", "")
        print(f"\n🧪 Evaluating test graph for disaster type: {dtype}")

        graph_path = os.path.join(graph_dir, fname)
        model_path = os.path.join(model_dir, f"gcn_{dtype}.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found for {dtype}, skipping.")
            continue

        # グラフとモデル読み込み
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

    # === 全体評価 ===
    if len(all_preds) == 0:
        print("⚠️ No predictions made.")
    else:
        print("\n📊 Overall GCN Test Evaluation (test-only graphs):")
        print("✅ Accuracy:", accuracy_score(all_labels, all_preds))
        print("✅ Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
        print("✅ Classification Report:\n",
              classification_report(all_labels, all_preds, target_names=["no-damage", "minor", "major", "destroyed"]))

# === 実行 ===
if __name__ == "__main__":
    evaluate()

# import os
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from tqdm import tqdm

# # === 入出力パス ===a
# graph_dir = "./graphs_with_image"
# model_dir = "./gcn_models_by_type"
# metadata_csv = "./all_buildings_for_graph.csv"
# test_image_list = "./split_lists/test_images.csv"

# # === GCNモデル定義（学習と一致させる） ===
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

# # === テスト対象ノード抽出 ===
# def get_test_mask(df_graph, test_ids):
#     def extract_image_id(path):
#         base = os.path.basename(path).replace(".tif", "")
#         parts = base.split("_")
#         return "_".join(parts[:2]) if len(parts) >= 2 else base
#     image_ids = df_graph["image_path"].apply(extract_image_id)
#     return image_ids.isin(test_ids)

# # === メイン評価関数 ===
# def evaluate_on_test(metadata_csv, test_image_list, graph_dir, model_dir):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     df_all = pd.read_csv(metadata_csv)

#     # ヘッダーを飛ばして読み込み
#     test_df = pd.read_csv(test_image_list)
#     test_ids = test_df["image_id"].dropna().str.lower().tolist()
#     if "image_id" in test_ids:
#         test_ids.remove("image_id")

#     print("📌 Test set contains {} unique image IDs".format(len(test_ids)))

#     all_preds = []
#     all_labels = []

#     for fname in os.listdir(graph_dir):
#         if not fname.endswith(".pt"): continue
#         dtype = fname.replace("graph_", "").replace(".pt", "")
#         print(f"\n🧪 Evaluating: {dtype}")

#         graph_path = os.path.join(graph_dir, fname)
#         model_path = os.path.join(model_dir, f"gcn_{dtype}.pt")
#         if not os.path.exists(model_path):
#             print(f"⚠️ Model not found for {dtype}, skipping.")
#             continue

#         # グラフ・モデル読み込み
#         g = torch.load(graph_path, weights_only=False)
#         model = GCNClassifier().to(device)
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.eval()

#         df_type = df_all[df_all["disaster_type"] == dtype].reset_index(drop=True)
#         mask = get_test_mask(df_type, test_ids)

#         print(f"📊 Test buildings in this type: {mask.sum()}")

#         if mask.sum() == 0:
#             print(f"⚠️ No test buildings for {dtype}, skipping.")
#             continue

#         x = g.x.to(device)
#         edge_index = g.edge_index.to(device)
#         y_true = g.y.cpu().numpy()
#         test_idx = torch.where(torch.tensor(mask.values))[0]

#         with torch.no_grad():
#             out = model(x, edge_index)
#             preds = out.argmax(dim=1).cpu().numpy()

#         y_pred = preds[test_idx]
#         y_gt = y_true[test_idx]

#         all_preds.extend(y_pred)
#         all_labels.extend(y_gt)

#         acc = accuracy_score(y_gt, y_pred)
#         print(f"✅ Accuracy for {dtype}: {acc:.4f}")

#     # === 全体評価 ===
#     print("\n📊 Overall GCN Test Evaluation:")
#     if len(all_preds) == 0:
#         print("⚠️ No test predictions available.")
#     else:
#         print("✅ Accuracy:", accuracy_score(all_labels, all_preds))
#         print("✅ Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
#         print("✅ Classification Report:\n",
#               classification_report(all_labels, all_preds, target_names=["no-damage", "minor", "major", "destroyed"]))

# # === 実行 ===
# if __name__ == "__main__":
#     evaluate_on_test(metadata_csv, test_image_list, graph_dir, model_dir)
