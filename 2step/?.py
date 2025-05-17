import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

# === Stage1 CNNモデル ===
class Stage1CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Linear(512, 2)  # binary: no-damage, damage

    def forward(self, x):
        return self.resnet(x)

# === Stage2 GATモデル ===
class Stage2GAT(nn.Module):
    def __init__(self, in_dim=518, hidden_dim=128, out_dim=3, heads=4, dropout=0.6):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, edge_index):
        from torch.nn import functional as F
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return self.mlp(x)

# === 画像読み込み ===
def load_image(path, size=224):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)).astype('float32') / 255.0
    img = torch.tensor(img).permute(2, 0, 1)
    return img

# === 実行関数 ===
def integrated_inference(metadata_csv, stage1_model_dir, stage2_graph_dir, device):
    df = pd.read_csv(metadata_csv)
    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    stage1 = Stage1CNN().to(device)
    stage1.load_state_dict(torch.load(os.path.join(stage1_model_dir, "stage1_cnn.pt"), map_location=device))
    stage1.eval()

    all_true = []
    all_pred = []

    for dtype in sorted(df["disaster_type"].unique()):
        print(f"\n🌍 Processing disaster type: {dtype}")
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)

        # Stage1 inference
        pred_stage1 = []
        for path in tqdm(df_type["image_path"], desc=f"Stage1: {dtype}"):
            img = load_image(path)
            if img is None:
                pred_stage1.append(0)  # fallback: no-damage
                continue
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = stage1(img)
                pred = out.argmax(dim=1).item()
            pred_stage1.append(pred)

        # Stage2 inference (on damage cases only)
        graph_path = os.path.join(stage2_graph_dir, f"graph_{dtype}.pt")
        g = torch.load(graph_path, map_location=device)
        stage2 = Stage2GAT().to(device)
        stage2.load_state_dict(torch.load(os.path.join(stage2_graph_dir, f"gat_{dtype}.pt"), map_location=device))
        stage2.eval()

        with torch.no_grad():
            out_stage2 = stage2(g.x.to(device), g.edge_index.to(device))
            pred_stage2 = out_stage2.argmax(dim=1).cpu().numpy()

        # Combine predictions
        final_pred = []
        idx = 0
        for p1 in pred_stage1:
            if p1 == 0:
                final_pred.append(0)  # no-damage
            else:
                final_pred.append(pred_stage2[idx] + 1)  # minor=1, major=2, destroyed=3
                idx += 1

        # True labels
        true_labels = df_type["label"].values
        all_true.extend(true_labels)
        all_pred.extend(final_pred)

    # Evaluation
    acc = accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred)
    report = classification_report(all_true, all_pred, target_names=["no-damage", "minor", "major", "destroyed"])

    print("\n✅ Overall Evaluation")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

# === 実行例 ===
if __name__ == "__main__":
    metadata_csv = "./cropped_building_metadata_with_shapes.csv"
    stage1_model_dir = "./stage1_models"
    stage2_graph_dir = "./stage2_graphs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrated_inference(metadata_csv, stage1_model_dir, stage2_graph_dir, device)
