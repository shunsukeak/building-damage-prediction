# === 修正: 災害名 → 災害タイプへのマッピング
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

# === Test推論メイン関数 (修正版)
def evaluate_test(test_image_list, model_dir, image_root, device):
    test_images = pd.read_csv(test_image_list)["image_id"].tolist()

    all_preds = []
    all_labels = []

    model_cache = {}

    for image_id in tqdm(test_images, desc="Testing"):
        pre_image_path = None
        for tier in ["tier1", "tier3", "test"]:
            candidate = os.path.join(image_root, tier, "images", f"{image_id}_pre_disaster.tif")
            if os.path.exists(candidate):
                pre_image_path = candidate
                break
        if pre_image_path is None:
            continue

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

                crop_img = crop_polygon_from_image(src, polygon)
                if crop_img is None:
                    continue

                # 🔵 ここで "災害タイプ" を推定する
                disaster_name = image_id.split("_")[0]
                disaster_type = DISASTER_NAME_TO_TYPE.get(disaster_name, "wildfire")  # デフォルトwildfireでもOK

                model_path = os.path.join(model_dir, f"model_type_{disaster_type}.pt")
                if disaster_type not in model_cache:
                    model = ResNetWithHazard()
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    model_cache[disaster_type] = model
                else:
                    model = model_cache[disaster_type]

                # 前処理
                img = cv2.resize(crop_img, (224, 224)).astype("float32") / 255.0
                img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
                hazard_level = get_hazard_level(disaster_name)  # これは元の災害名ベースで良い
                hazard_level = torch.tensor([[hazard_level]], dtype=torch.float32).to(device)

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

# === 実行例（修正版）
if __name__ == "__main__":
    test_image_list = "./split_lists/test_images.csv"
    model_dir = "./saved_type_models"  # 🔵 ここもtypeモデル用
    image_root = "./data/geotiffs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_test(test_image_list, model_dir, image_root, device)
