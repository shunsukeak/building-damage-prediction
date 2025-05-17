import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon
from tqdm import tqdm

input_csv = "../csv/cropped_building_metadata_with_shapes.csv" 
output_csv = "../csv/all_buildings_for_graph.csv"  

def generate_graph_metadata(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    xs, ys = [], []

    for poly_str in tqdm(df["geometry"], desc="Extracting centroid"):
        try:
            poly = wkt.loads(poly_str)
            centroid = poly.centroid
            xs.append(centroid.x)
            ys.append(centroid.y)
        except:
            xs.append(None)
            ys.append(None)

    df["x"] = xs
    df["y"] = ys

    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print(f"✅ Saved graph metadata: {output_csv}")

if __name__ == "__main__":
    generate_graph_metadata(input_csv, output_csv)
