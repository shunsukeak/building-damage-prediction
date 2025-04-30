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
graph_dir = "./graphs_by_type"
metadata_csv = "./all_buildings_for_graph.csv"
output_dir = "./graphs_with_image"
os.makedirs(output_dir, exist_ok=True)

# === ResNet feature extractor ===
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet34(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # (B, 512)

# === 画像を読み込んでTensorにする ===
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
        return None

# === 実行 ===
def extract_and_merge(graph_dir, metadata_csv, output_dir):
    df = pd.read_csv(metadata_csv)
    resnet = ResNetEncoder().eval().cuda()
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    for fname in os.listdir(graph_dir):
        if not fname.endswith(".pt"): continue
        dtype = fname.replace("graph_", "").replace(".pt", "")
        print(f"🔍 Processing: {dtype}")

        # 該当グラフとノードに対応するメタデータ
        g = torch.load(os.path.join(graph_dir, fname))
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
        assert len(df_type) == g.num_nodes, f"Mismatch in node count: {dtype}"

        # 画像パスをすべて読み込んでResNetに通す
        all_imgs = []
        for path in df_type["image_path"]:
            img = load_and_preprocess_image(path)
            if img is None:
                img = torch.zeros(3, 224, 224)  # 空白画像で代替
            all_imgs.append(transform(img).unsqueeze(0))

        img_batch = torch.cat(all_imgs, dim=0).cuda()  # (N, 3, 224, 224)

        with torch.no_grad():
            img_feats = resnet(img_batch).cpu()  # (N, 512)

        # 元の x（shape+hazard）とconcat
        x_shape = g.x  # (N, 6)
        x_combined = torch.cat([img_feats, x_shape], dim=1)  # (N, 518)

        # 再構築
        new_g = Data(x=x_combined, edge_index=g.edge_index, y=g.y)
        new_g.disaster_type = dtype

        save_path = os.path.join(output_dir, f"graph_{dtype}.pt")
        torch.save(new_g, save_path)
        print(f"✅ Saved: {save_path} ({x_combined.shape[0]} nodes)")

# === 実行 ===
if __name__ == "__main__":
    extract_and_merge(graph_dir, metadata_csv, output_dir)
