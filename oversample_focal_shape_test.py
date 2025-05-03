# oversample_focal_shape_test.py
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import rasterio
from rasterio import mask
from shapely.geometry import mapping, box
from shapely import wkt
from shapely.ops import transform
import pyproj
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === 再投影関数 ===
def reproject_polygon(polygon, from_crs, to_crs):
    project = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True).transform
    return transform(project, polygon)

# === crop関数 ===
def crop_polygon_from_image(src, polygon):
    try:
        if not polygon.is_valid:
            return None
        reprojected = reproject_polygon(polygon, "EPSG:4326", src.crs)
        if not reprojected.intersects(box(*src.bounds)):
            return None
        out_image, _ = mask.mask(src, [mapping(reprojected)], crop=True)
        img = out_image.transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = img.repeat(3, axis=2)
        return img
    except:
        return None

# === モデル定義 ===
class ResNetWithHazardAndShape(nn.Module):
    def __init__(self, num_classes=4):
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

# === マッピングと定数 ===
def label_to_int(subtype):
    return {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}.get(subtype, -1)

def get_hazard_level(disaster_name):
    return {
        "mexico-earthquake": 5,
        "santa-rosa-wildfire": 4,
        "pinery-bushfire": 3,
        "portugal-wildfire": 4,
        "woolsey-fire": 4,
        "socal-fire": 4,
        "midwest-flooding": 4,
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
    }.get(disaster_name, 3)

DISASTER_NAME_TO_TYPE = {
    "mexico-earthquake": "earthquake",
    "santa-rosa-wildfire": "wildfire",
    "pinery-bushfire": "wildfire",
    "portugal-wildfire": "wildfire",
    "woolsey-fire": "wildfire",
    "socal-fire": "wildfire",
    "midwest-flooding": "flood",
    "nepal-flooding": "flood",
    "hurricane-florence": "hurricane",
    "hurricane-harvey": "hurricane",
    "hurricane-matthew": "hurricane",
    "hurricane-michael": "hurricane",
    "joplin-tornado": "tornado",
    "moore-tornado": "tornado",
    "tuscaloosa-tornado": "tornado",
    "palu-tsunami": "tsunami",
    "sunda-tsunami": "tsunami",
    "guatemala-volcano": "volcanic_eruption",
    "lower-puna-volcano": "volcanic_eruption"
}

# === ポリゴンからshape featuresを計算 ===
def compute_shape_features(poly):
    try:
        area = poly.area
        peri = poly.length
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        aspect = w / h if h > 0 else 0
        extent = area / (w * h) if w * h > 0 else 0
        convex = area / poly.convex_hull.area if poly.convex_hull.area > 0 else 0
        return [area, peri, aspect, extent, convex]
    except:
        return [0, 0, 0, 0, 0]

# === テスト実行関数 ===
def evaluate_test(test_image_list, model_dir, image_root, device):
    test_images = pd.read_csv(test_image_list)["image_id"].tolist()

    all_preds, all_labels, all_image_ids = [], [], []
    image_pred_dict = defaultdict(list)
    image_label_dict = defaultdict(list)
    model_cache = {}

    disaster_correct = defaultdict(int)
    disaster_total = defaultdict(int)
    disaster_name_pred_dict = defaultdict(list)
    disaster_name_label_dict = defaultdict(list)

    for image_id in tqdm(test_images, desc="Testing"):
        pre_path, label_path = None, None
        for tier in ["tier1", "tier3", "test"]:
            img_path = os.path.join(image_root, tier, "images", f"{image_id}_pre_disaster.tif")
            lbl_path = os.path.join(image_root, tier, "labels", f"{image_id}_post_disaster.json")
            if os.path.exists(img_path): pre_path = img_path
            if os.path.exists(lbl_path): label_path = lbl_path
        if not pre_path or not label_path:
            continue

        with rasterio.open(pre_path) as src, open(label_path, "r") as f:
            data = json.load(f)
            for feat in data.get("features", {}).get("lng_lat", []):
                subtype = feat["properties"].get("subtype")
                if subtype not in ["no-damage", "minor-damage", "major-damage", "destroyed"]:
                    continue
                try:
                    poly = wkt.loads(feat["wkt"])
                    crop = crop_polygon_from_image(src, poly)
                    if crop is None:
                        continue
                except:
                    continue

                disaster_name = image_id.split("_")[0]
                disaster_type = DISASTER_NAME_TO_TYPE.get(disaster_name, "wildfire")

                if disaster_type not in model_cache:
                    model_path = os.path.join(model_dir, f"model_type_{disaster_type}.pt")
                    model = ResNetWithHazardAndShape()
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    model_cache[disaster_type] = model
                else:
                    model = model_cache[disaster_type]

                img = cv2.resize(crop, (224, 224)).astype("float32") / 255.0
                img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
                hazard = get_hazard_level(disaster_name)
                hazard = torch.tensor([[hazard]], dtype=torch.float32).to(device)
                shape_feat = torch.tensor([compute_shape_features(poly)], dtype=torch.float32).to(device)

                with torch.no_grad():
                    output = model(img, hazard, shape_feat)
                    pred = output.argmax(dim=1).item()
                    label = label_to_int(subtype)

                all_preds.append(pred)
                all_labels.append(label)
                all_image_ids.append(image_id)
                image_pred_dict[image_id].append(pred)
                image_label_dict[image_id].append(label)
                disaster_name_pred_dict[disaster_name].append(pred)
                disaster_name_label_dict[disaster_name].append(label)

                # === 災害タイプごとの集計
                if pred == label:
                    disaster_correct[disaster_type] += 1
                disaster_total[disaster_type] += 1

    # === 全体評価
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["no-damage", "minor", "major", "destroyed"])

    print(f"\n✅ Test Accuracy (building-level): {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    print("\n🖼️ Image-level Accuracy:")
    accs = []
    for img_id in sorted(image_pred_dict.keys()):
        preds = np.array(image_pred_dict[img_id])
        labels = np.array(image_label_dict[img_id])
        image_acc = (preds == labels).sum() / len(labels)
        accs.append(image_acc)
        print(f"🖼️ {img_id}: {image_acc:.2f} ({(preds==labels).sum()}/{len(labels)})")

    print(f"\n📈 Mean Image-level Accuracy: {np.mean(accs):.4f}")

    # === 災害タイプごとのaccuracy
    print("\n🌎 Disaster Type-wise Accuracy:")
    for dtype in sorted(disaster_total.keys()):
        correct = disaster_correct[dtype]
        total = disaster_total[dtype]
        type_acc = correct / total if total > 0 else 0
        print(f"🌍 {dtype}: {type_acc:.4f} ({correct}/{total})")

    # === 災害名単位集計 ===
    print("\n🌋 Disaster-name summary:")
    disaster_name_accs = []
    for dname in sorted(disaster_name_pred_dict.keys()):
        preds = np.array(disaster_name_pred_dict[dname])
        labels = np.array(disaster_name_label_dict[dname])
        correct = (preds == labels).sum()
        total = len(labels)
        dname_acc = correct / total
        disaster_name_accs.append(dname_acc)
        print(f"🌋 {dname}: Accuracy = {dname_acc:.4f} ({correct}/{total})")
    mean_dname_acc = np.mean(disaster_name_accs)
    print(f"\n📊 Mean Disaster-name Accuracy: {mean_dname_acc:.4f}")
# === 実行例 ===
if __name__ == "__main__":
    test_image_list = "./split_lists/test_images.csv"
    model_dir = "./oversample_focal_saved_type_models_with_shape"
    image_root = "./data/geotiffs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_test(test_image_list, model_dir, image_root, device)

