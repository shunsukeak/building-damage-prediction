# 2_cnn_inference.py

import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# === Dataset ===
class CropBuildingDatasetWithShapeBinary(Dataset):
    def __init__(self, csv_path, transform=None, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df["image_path"].tolist()
        self.hazard_levels = self.df["hazard_level"].tolist()
        self.shape_features = self.df[["area", "perimeter", "aspect_ratio", "extent_ratio", "convexity"]].fillna(0).values.astype("float32")
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype("float32") / 255.0
        return img

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        hazard_level = torch.tensor([self.hazard_levels[idx]], dtype=torch.float32)
        shape_feat = torch.tensor(self.shape_features[idx], dtype=torch.float32)
        img = torch.tensor(img).permute(2, 0, 1)
        return img, hazard_level, shape_feat, self.image_paths[idx]

# === model ===
class ResNetWithHazardAndShape(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet34(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()

        self.hazard_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.shape_fc = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 32 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_hazard, x_shape):
        img_feat = self.resnet(x_img)
        hazard_feat = self.hazard_fc(x_hazard)
        shape_feat = self.shape_fc(x_shape)
        feat = torch.cat([img_feat, hazard_feat, shape_feat], dim=1)
        return self.classifier(feat)

# === inference ===
def run_inference(test_csv, model_dir, output_csv, damage_csv, device, batch_size=32):
    df = pd.read_csv(test_csv)
    disaster_types = df["disaster_type"].unique()
    all_results = []

    for dtype in disaster_types:
        print(f"🚀 Inference on disaster type: {dtype}")
        df_type = df[df["disaster_type"] == dtype].reset_index(drop=True)
        test_dataset = CropBuildingDatasetWithShapeBinary(df_type)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model_path = os.path.join(model_dir, f"best_model_type_{dtype}.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found for {dtype}, skipping.")
            continue

        model = ResNetWithHazardAndShape().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preds = []
        image_paths = []

        with torch.no_grad():
            for img, hazard, shape, paths in tqdm(test_loader, desc=f"{dtype} inference"):
                img, hazard, shape = img.to(device), hazard.to(device), shape.to(device)
                output = model(img, hazard, shape)
                pred = output.argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                image_paths.extend(paths)

        # Save results
        df_type["pred_binary_label"] = preds
        all_results.append(df_type)

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(output_csv, index=False)
    print(f"✅ Saved all predictions to {output_csv}")

    # Damage
    df_damage = df_all[df_all["pred_binary_label"] == 1].reset_index(drop=True)
    df_damage.to_csv(damage_csv, index=False)
    print(f"✅ Saved damage-only records to {damage_csv}")


if __name__ == "__main__":
    test_csv = "../all_buildings_for_graph.csv"
    model_dir = "../binary_cnn_models"
    output_csv = "../stage1_predictions.csv"
    damage_csv = "../stage1_damage_only.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_inference(test_csv, model_dir, output_csv, damage_csv, device)
