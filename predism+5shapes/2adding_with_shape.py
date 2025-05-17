# 2-3 adding shape features
import pandas as pd
from shapely import wkt
from tqdm import tqdm

input_csv = "../csv/cropped_building_metadata_with_preshape.csv"
output_csv = "../csv/cropped_building_metadata_with_shapes.csv"

# === shape feature calculation ===
def compute_shape_features(wkt_str):
    try:
        poly = wkt.loads(wkt_str)
        if not poly.is_valid:
            return [0, 0, 0, 0, 0]
        
        area = poly.area
        perimeter = poly.length

        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny
        aspect_ratio = width / height if height != 0 else 0

        bbox_area = width * height
        extent_ratio = area / bbox_area if bbox_area != 0 else 0

        convex_hull_area = poly.convex_hull.area
        convexity = area / convex_hull_area if convex_hull_area != 0 else 0

        return [area, perimeter, aspect_ratio, extent_ratio, convexity]
    except Exception as e:
        return [0, 0, 0, 0, 0]

def add_shape_features(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    areas, perimeters, aspect_ratios, extent_ratios, convexities = [], [], [], [], []

    for wkt_str in tqdm(df["geometry"], desc="Computing shape features"):
        area, perimeter, aspect_ratio, extent_ratio, convexity = compute_shape_features(wkt_str)
        areas.append(area)
        perimeters.append(perimeter)
        aspect_ratios.append(aspect_ratio)
        extent_ratios.append(extent_ratio)
        convexities.append(convexity)

    # new columns
    df["area"] = areas
    df["perimeter"] = perimeters
    df["aspect_ratio"] = aspect_ratios
    df["extent_ratio"] = extent_ratios
    df["convexity"] = convexities

    df.to_csv(output_csv, index=False)
    print(f"✅ Saved metadata with shape features: {output_csv}")

if __name__ == "__main__":
    add_shape_features(input_csv, output_csv)
