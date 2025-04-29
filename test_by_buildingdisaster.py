# 1-5. test
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import cv2
import rasterio
from rasterio import mask
from shapely.geometry import mapping, box
from shapely import wkt
from shapely.ops import transform
import pyproj
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === reproject_polygon ===
def reproject_polygon(polygon, from_crs, to_crs):
    project = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True).transform
    return transform(project, polygon)

# === crop_polygon_from_image (オンザフライcrop) ===
def crop_polygon_from_image(src, polygon):
    try:
        if not polygon.is_valid:
            return None

        reprojected_polygon = reproject_polygon(polygon, "EPSG:4326", src.crs)
        img_bounds = box(*src.bounds)
        if not reprojected_polygon.intersects(img_bounds):
            return None

        out_image, _ = mask.mask(src, [mapping(reprojected_polygon)], crop=True)
        img = out_image.transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = img.repeat(3, axis=2)
        return img
    except:
        return None

# === ResNet with Hazard (Same as Training) ===
class ResNetWithHazard(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.resnet = models.resnet34(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()
        self.meta_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_meta):
        img_feat = self.resnet(x_img)
        meta_feat = self.meta_fc(x_meta)
        feat = torch.cat([img_feat, meta_feat], dim=1)
        out = self.classifier(feat)
        return out

# === Test推論メイン関数 ===
def evaluate_test(test_image_list, model_dir, label_dirs, image_root, device):
    test_images = pd.read_csv(test_image_list)["image_id"].tolist()

    all_preds = []
    all_labels = []

    model_cache = {}

    for image_id in tqdm(test_images, desc="Testing"):
        # pre-disaster image path
        pre_image_path = None
        for tier in ["tier1", "tier3", "test"]:
            candidate = os.path.join(image_root, tier, "images", f"{image_id}_pre_disaster.tif")
            if os.path.exists(candidate):
                pre_image_path = candidate
                break
        if pre_image_path is None:
            continue

        # label file path
        label_path = None
        for tier in ["tier1", "tier3", "test"]:
            candidate = os.path.join(image_root, tier, "labels", f"{image_id}_post_disaster.json")
            if os.path.exists(candidate):
                label_path = candidate
                break
        if label_path is None:
            continue

        with rasterio.open(pre_image_path) as src:
            with open(label_path, "r") as f:
                label_data = json.load(f)

            for feature in label_data.get("features", {}).get("lng_lat", []):
                subtype = feature["properties"].get("subtype")
                if subtype not in ["no-damage", "minor-damage", "major-damage", "destroyed"]:
                    continue

                try:
                    polygon = wkt.loads(feature["wkt"])
                except:
                    continue

                # 建物crop
                crop_img = crop_polygon_from_image(src, polygon)
                if crop_img is None:
                    continue

                # 災害名からモデルを選ぶ
                disaster_name = image_id.split("_")[0]  # 例: hurricane-harvey
                model_path = os.path.join(model_dir, f"model_{disaster_name}.pt")
                if disaster_name not in model_cache:
                    model = ResNetWithHazard()
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    model_cache[disaster_name] = model
                else:
                    model = model_cache[disaster_name]

                # 前処理
                img = cv2.resize(crop_img, (224, 224)).astype("float32") / 255.0
                img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
                hazard_level = get_hazard_level(disaster_name)  # 後述
                hazard_level = torch.tensor([[hazard_level]], dtype=torch.float32).to(device)

                # 推論
                with torch.no_grad():
                    output = model(img, hazard_level)
                    pred = output.argmax(dim=1).item()

                label = label_to_int(subtype)

                all_preds.append(pred)
                all_labels.append(label)

    # 結果集計
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["no-damage", "minor", "major", "destroyed"])

    print(f"✅ Test Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

# === ヘルパー関数 ===
def label_to_int(subtype):
    mapping = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
    return mapping.get(subtype, -1)

def get_hazard_level(disaster_name):
    HAZARD_LEVEL_MAP = {
        "mexico-city-earthquake": 5,
        "santa-rosa-wildfire": 4,
        "pinery-wildfire": 3,
        "portugal-wildfire": 4,
        "woolsey-fire": 4,
        "midwest-floods": 4,
        "nepal-flooding": 3,
        "hurricane-florence": 5,
        "hurricane-harvey": 5,
        "hurricane-matthew": 4,
        "hurricane-michael": 5,
        "joplin-tornado": 5,
        "moore-tornado": 5,
        "tuscaloosa-tornado": 5,
        "palu-tsunami": 4,
        "sunda-tsunami": 3,
        "guatemala-volcano": 4,
        "lower-puna-volcano": 3
    }
    return HAZARD_LEVEL_MAP.get(disaster_name, 3)

# === 実行例 ===
if __name__ == "__main__":
    test_image_list = "./split_lists/test_images.csv"
    model_dir = "./saved_crop_models"
    image_root = "./data/geotiffs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_test(test_image_list, model_dir, label_dirs=None, image_root=image_root, device=device)
