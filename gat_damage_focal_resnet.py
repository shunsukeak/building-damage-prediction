# gat_damage_focal_resnet.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from torch_geometric.data import Data
from tqdm import tqdm

# === 入出力 ===
graph_dir = "./graphs_by_type_damage"
metadata_csv = "./all_buildings_for_graph.csv"
output_dir = "./graphs_with_image_damage"
os.makedirs(output_dir, exist_ok=True)

# === ResNet feature extractor ===
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet34(weights="IMAGENETK_V1")
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # (B, 512)

# === 画像読み込み（エラーはzero tensorで代用） ===
def load_and_preprocess_image(path, image_size=224):
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size)).astype("float32") / 255.0
        img = torch.tensor(img).permute(2, 0, 1)
        return img
    except:
        return torch.zeros(3, image_size, image_size)

# === 実行 ===
def extract_and_merge(graph_dir, metadata_csv, output_dir):
    df = pd.read_csv(metadata_csv)
    resnet = ResNetEncoder().eval().cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    for fname in os.listdir(graph_dir):
        if not fname.endswith(".pt"): continue
        dtype = fname.replace("graph_", "").replace(".pt", "")
        print(f"🔍 Processing: {dtype}")

        g = torch.load(os.path.join(graph_dir, fname))
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
        assert len(df_type) == g.num_nodes, f"Node count mismatch in {dtype}"

        img_feats = []

        for path in tqdm(df_type["image_path"], desc=f"Extracting features ({dtype})"):
            img = load_and_preprocess_image(path)
            img = normalize(img).unsqueeze(0).cuda()
            with torch.no_grad():
                feat = resnet(img).cpu()
            img_feats.append(feat)

        img_feats = torch.cat(img_feats, dim=0)  # (N, 512)
        x_combined = torch.cat([img_feats, g.x], dim=1)  # (N, 512 + 9 = 521)

        new_g = Data(x=x_combined, edge_index=g.edge_index, y=g.y)
        new_g.disaster_type = dtype

        save_path = os.path.join(output_dir, f"graph_{dtype}.pt")
        torch.save(new_g, save_path)
        print(f"✅ Saved: {save_path} ({x_combined.shape[0]} nodes)")

# === 実行例 ===
if __name__ == "__main__":
    extract_and_merge(graph_dir, metadata_csv, output_dir)
