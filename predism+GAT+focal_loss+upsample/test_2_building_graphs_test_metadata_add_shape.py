from shapely import wkt
import pandas as pd

def add_shape_features(df):
    areas, peris, ars, exts, convs = [], [], [], [], []
    for w in df["geometry"]:
        try:
            poly = wkt.loads(w)
            area = poly.area
            peri = poly.length
            bounds = poly.bounds
            w_, h_ = bounds[2] - bounds[0], bounds[3] - bounds[1]
            ar = w_ / h_ if h_ != 0 else 0
            ext = area / (w_ * h_) if w_ * h_ != 0 else 0
            conv = area / poly.convex_hull.area if poly.convex_hull.area != 0 else 0
        except:
            area, peri, ar, ext, conv = [0]*5
        areas.append(area)
        peris.append(peri)
        ars.append(ar)
        exts.append(ext)
        convs.append(conv)

    df["area"] = areas
    df["perimeter"] = peris
    df["aspect_ratio"] = ars
    df["extent_ratio"] = exts
    df["convexity"] = convs
    return df

df = pd.read_csv("../csv/cropped_test_building_metadata.csv")
df = add_shape_features(df)
df.to_csv("../csv/cropped_test_building_metadata_with_shape.csv", index=False)
