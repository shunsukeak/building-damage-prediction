import os
import json
import pandas as pd
import rasterio
from rasterio import mask
from shapely import wkt
from tqdm import tqdm

# === パラメータ設定 ===
label_dirs = [
    "./data/geotiffs/tier1/labels",
    "./data/geotiffs/tier3/labels",
    "./data/geotiffs/test/labels"
]
image_root = "./data/geotiffs"
train_image_list = "./split_lists/train_images.csv"
output_crop_dir = "./data/cropped_buildings"
output_crop_metadata = "./cropped_building_metadata.csv"

DISASTER_NAME_TO_TYPE = {
    "mexico-city-earthquake": "earthquake",
    "santa-rosa-wildfire": "wildfire",
    "pinery-wildfire": "wildfire",
    "portugal-wildfire": "wildfire",
    "woolsey-fire": "wildfire",
    "midwest-floods": "flood",
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

LABEL_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

# === crop_polygon_from_image (xy版) ===
def crop_polygon_from_image(src, polygon, out_path):
    try:
        out_image, out_transform = mask.mask(src, [polygon], crop=True)
        if out_image.shape[1] == 0 or out_image.shape[2] == 0:
            print(f"⚠️ Skipping empty crop for {out_path}")
            return False

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)
        print(f"✅ Cropped: {out_path}")
        return True
    except Exception as e:
        print(f"⚠️ Error cropping: {e}")
        return False
# === メイン処理 ===
def extract_crop_buildings(label_dirs, image_root, train_image_list, output_crop_dir, output_crop_metadata):
    os.makedirs(output_crop_dir, exist_ok=True)

    train_images = pd.read_csv(train_image_list)["image_id"].tolist()

    records = []

    for label_dir in label_dirs:
        for fname in tqdm(os.listdir(label_dir), desc=f"Processing {label_dir}"):
            if not fname.endswith(".json"):
                continue
            disaster_name = fname.replace("_post_disaster.json", "").lower()
            if disaster_name not in train_images:
                continue

            fpath = os.path.join(label_dir, fname)
            pre_image = fname.replace("_post_disaster.json", "_pre_disaster.tif")
            pre_image_path = None
            for tier in ["tier1", "tier3", "test"]:
                candidate = os.path.join(image_root, tier, "images", pre_image)
                if os.path.exists(candidate):
                    pre_image_path = candidate
                    break
            if pre_image_path is None:
                continue

            disaster_type = DISASTER_NAME_TO_TYPE.get(disaster_name)
            hazard_level = HAZARD_LEVEL_MAP.get(disaster_name)
            if disaster_type is None or hazard_level is None:
                continue

            with open(fpath, "r") as f:
                data = json.load(f)

            xy_features = data.get("features", {}).get("xy", [])
            if not xy_features:
                continue

            with rasterio.open(pre_image_path) as src:
                for i, feature in enumerate(xy_features):
                    subtype = feature["properties"].get("subtype")
                    print(f"Found subtype: {subtype}")
                    if subtype not in LABEL_MAP:
                        print(f"⚠️ Skipped because unknown subtype: {subtype}")
                        continue
                    try:
                        polygon = wkt.loads(feature["wkt"])
                    except:
                        continue

                    out_file = f"{disaster_name}_{i}.tif"
                    out_path = os.path.join(output_crop_dir, out_file)

                    success = crop_polygon_from_image(src, mapping(polygon), out_path)
                    if success:
                        records.append({
                            "image_path": out_path,
                            "label": LABEL_MAP[subtype],
                            "disaster_type": disaster_type,
                            "hazard_level": hazard_level,
                            "disaster_name": disaster_name
                        })

    df = pd.DataFrame(records)
    df.to_csv(output_crop_metadata, index=False)
    print(f"✅ Saved crop metadata: {output_crop_metadata}")

# === 実行例 ===
if __name__ == "__main__":
    extract_crop_buildings(label_dirs, image_root, train_image_list, output_crop_dir, output_crop_metadata)

# import os
# import json
# import pandas as pd
# import rasterio
# from rasterio import mask
# from shapely.geometry import box, mapping
# from shapely import wkt
# from shapely.ops import transform
# import pyproj
# from tqdm import tqdm

# # === パラメータ設定 ===
# label_dirs = [
#     "./data/geotiffs/tier1/labels",
#     "./data/geotiffs/tier3/labels",
#     "./data/geotiffs/test/labels"
# ]
# image_root = "./data/geotiffs"
# train_image_list = "./split_lists/train_images.csv"
# output_crop_dir = "./data/cropped_buildings"
# output_crop_metadata = "./cropped_building_metadata.csv"

# DISASTER_NAME_TO_TYPE = {
#     "mexico-city-earthquake": "earthquake",
#     "santa-rosa-wildfire": "wildfire",
#     "pinery-wildfire": "wildfire",
#     "portugal-wildfire": "wildfire",
#     "woolsey-fire": "wildfire",
#     "midwest-floods": "flood",
#     "nepal-flooding": "flood",
#     "hurricane-florence": "hurricane",
#     "hurricane-harvey": "hurricane",
#     "hurricane-matthew": "hurricane",
#     "hurricane-michael": "hurricane",
#     "joplin-tornado": "tornado",
#     "moore-tornado": "tornado",
#     "tuscaloosa-tornado": "tornado",
#     "palu-tsunami": "tsunami",
#     "sunda-tsunami": "tsunami",
#     "guatemala-volcano": "volcanic_eruption",
#     "lower-puna-volcano": "volcanic_eruption"
# }

# HAZARD_LEVEL_MAP = {
#     "mexico-city-earthquake": 5,
#     "santa-rosa-wildfire": 4,
#     "pinery-wildfire": 3,
#     "portugal-wildfire": 4,
#     "woolsey-fire": 4,
#     "midwest-floods": 4,
#     "nepal-flooding": 3,
#     "hurricane-florence": 5,
#     "hurricane-harvey": 5,
#     "hurricane-matthew": 4,
#     "hurricane-michael": 5,
#     "joplin-tornado": 5,
#     "moore-tornado": 5,
#     "tuscaloosa-tornado": 5,
#     "palu-tsunami": 4,
#     "sunda-tsunami": 3,
#     "guatemala-volcano": 4,
#     "lower-puna-volcano": 3
# }

# LABEL_MAP = {
#     "no-damage": 0,
#     "minor-damage": 1,
#     "major-damage": 2,
#     "destroyed": 3
# }

# # === reproject_polygon関数 ===
# def reproject_polygon(polygon, from_crs, to_crs):
#     project = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True).transform
#     return transform(project, polygon)

# # === crop_polygon_from_image関数 ===
# def crop_polygon_from_image(image_path, polygon, out_path):
#     with rasterio.open(image_path) as src:
#         try:
#             if not polygon.is_valid:
#                 return False

#             reprojected_polygon = reproject_polygon(polygon, "EPSG:4326", src.crs)

#             img_bounds = box(*src.bounds)
#             if not reprojected_polygon.intersects(img_bounds):
#                 return False

#             out_image, out_transform = mask.mask(src, [mapping(reprojected_polygon)], crop=True)
#             out_meta = src.meta.copy()
#             out_meta.update({
#                 "driver": "GTiff",
#                 "height": out_image.shape[1],
#                 "width": out_image.shape[2],
#                 "transform": out_transform
#             })

#             with rasterio.open(out_path, "w", **out_meta) as dest:
#                 dest.write(out_image)
#             return True
#         except Exception as e:
#             print(f"⚠️ Error cropping: {e}")
#             return False

# # === 建物cropとmetadata作成 ===
# def extract_crop_buildings(label_dirs, image_root, train_image_list, output_crop_dir, output_crop_metadata):
#     os.makedirs(output_crop_dir, exist_ok=True)

#     train_images = pd.read_csv(train_image_list)["image_id"].tolist()

#     records = []

#     for label_dir in label_dirs:
#         for fname in tqdm(os.listdir(label_dir), desc=f"Processing {label_dir}"):
#             if not fname.endswith(".json"):
#                 continue
#             disaster_name = fname.replace("_post_disaster.json", "").lower()
#             if disaster_name not in train_images:
#                 continue

#             fpath = os.path.join(label_dir, fname)
#             pre_image = fname.replace("_post_disaster.json", "_pre_disaster.tif")
#             pre_image_path = None
#             for tier in ["tier1", "tier3", "test"]:
#                 candidate = os.path.join(image_root, tier, "images", pre_image)
#                 if os.path.exists(candidate):
#                     pre_image_path = candidate
#                     break
#             if pre_image_path is None:
#                 continue

#             disaster_type = DISASTER_NAME_TO_TYPE.get(disaster_name)
#             hazard_level = HAZARD_LEVEL_MAP.get(disaster_name)
#             if disaster_type is None or hazard_level is None:
#                 continue

#             with open(fpath, "r") as f:
#                 data = json.load(f)

#             for i, feature in enumerate(data.get("features", {}).get("lng_lat", [])):
#                 subtype = feature["properties"].get("subtype")
#                 if subtype not in LABEL_MAP:
#                     continue
#                 try:
#                     polygon = wkt.loads(feature["wkt"])
#                 except:
#                     continue

#                 out_file = f"{disaster_name}_{i}.tif"
#                 out_path = os.path.join(output_crop_dir, out_file)

#                 success = crop_polygon_from_image(pre_image_path, polygon, out_path)
#                 if success:
#                     records.append({
#                         "image_path": out_path,
#                         "label": LABEL_MAP[subtype],
#                         "disaster_type": disaster_type,
#                         "hazard_level": hazard_level,
#                         "disaster_name": disaster_name
#                     })

#     df = pd.DataFrame(records)
#     df.to_csv(output_crop_metadata, index=False)
#     print(f"✅ Saved crop metadata: {output_crop_metadata}")

# # === 実行例 ===
# if __name__ == "__main__":
#     extract_crop_buildings(label_dirs, image_root, train_image_list, output_crop_dir, output_crop_metadata)
