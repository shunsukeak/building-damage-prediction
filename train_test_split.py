# 1-1, 2-1 Train_test split
import os
import random
import pandas as pd

# === パラメータ設定 ===
pre_image_root = "./data/geotiffs"  # pre-disaster画像があるtier1/tier3のroot
output_dir = "./split_lists"
train_ratio = 0.8
random_seed = 42

# === 画像一覧を取得し、train/testに分割 ===
def split_images(pre_image_root, output_dir, train_ratio=0.8, random_seed=42):
    pre_images = []
    for tier in ["tier1", "tier3", "test"]:
        image_dir = os.path.join(pre_image_root, tier, "images")
        for fname in os.listdir(image_dir):
            if fname.endswith("_pre_disaster.tif"):
                image_id = fname.replace("_pre_disaster.tif", "").lower()
                pre_images.append(image_id)

    print(f"✅ Found {len(pre_images)} pre-disaster images.")

    random.seed(random_seed)
    random.shuffle(pre_images)

    split_idx = int(len(pre_images) * train_ratio)
    train_images = pre_images[:split_idx]
    test_images = pre_images[split_idx:]

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({"image_id": train_images}).to_csv(os.path.join(output_dir, "train_images.csv"), index=False)
    pd.DataFrame({"image_id": test_images}).to_csv(os.path.join(output_dir, "test_images.csv"), index=False)

    print(f"✅ Train images: {len(train_images)}, Test images: {len(test_images)}")

# === 実行例 ===
if __name__ == "__main__":
    split_images(pre_image_root, output_dir, train_ratio, random_seed)
