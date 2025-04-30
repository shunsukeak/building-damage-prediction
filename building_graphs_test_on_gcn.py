import os
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

# === 入出力パス ===
graph_dir = "./graphs_with_image"
model_dir = "./gcn_models_by_type"
metadata_csv = "./all_buildings_for_graph.csv"
test_image_list = "./split_lists/test_images.csv"

# === GCNモデル定義（学習と一致させる） ===
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

# === テスト対象ノードの抽出関数 ===
def get_test_mask(df_graph, df_test_ids):
    image_ids = df_graph["image_path"].apply(lambda x: os.path.basename(x).replace("_pre_disaster.tif", "").split(".")[0].lower())
    return image_ids.isin(df_test_ids)

# === 実行 ===
def evaluate_on_test(metadata_csv, test_image_list, graph_dir, model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_all = pd.read_csv(metadata_csv)
    test_df = pd.read_csv(test_image_list)
    test_ids = test_df["image_id"].str.lower().tolist()

    all_preds = []
    all_labels = []

    for fname in os.listdir(graph_dir):
        if not fname.endswith(".pt"): continue
        dtype = fname.replace("graph_", "").replace(".pt", "")
        print(f"\n🧪 Evaluating: {dtype}")

        graph_path = os.path.join(graph_dir, fname)
        model_path = os.path.join(model_dir, f"gcn_{dtype}.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found for {dtype}, skipping.")
            continue

        # グラフ・モデル読み込み
        g = torch.load(graph_path)
        model = GCNClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 該当するノードを絞り込む
        df_type = df_all[df_all["disaster_type"] == dtype].reset_index(drop=True)
        mask = get_test_mask(df_type, test_ids)

        if mask.sum() == 0:
            print(f"⚠️ No test buildings for {dtype}, skipping.")
            continue

        x = g.x.to(device)
        edge_index = g.edge_index.to(device)
        y_true = g.y.cpu().numpy()
        test_idx = torch.where(torch.tensor(mask.values))[0]

        with torch.no_grad():
            out = model(x, edge_index)
            preds = out.argmax(dim=1).cpu().numpy()

        # 評価対象のみ抽出
        y_pred = preds[test_idx]
        y_gt = y_true[test_idx]

        all_preds.extend(y_pred)
        all_labels.extend(y_gt)

        acc = accuracy_score(y_gt, y_pred)
        print(f"✅ Accuracy for {dtype}: {acc:.4f}")

    # === 全体評価 ===
    print("\n📊 Overall GCN Test Evaluation:")
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["no-damage", "minor", "major", "destroyed"]))

# === 実行 ===
if __name__ == "__main__":
    evaluate_on_test(metadata_csv, test_image_list, graph_dir, model_dir)
