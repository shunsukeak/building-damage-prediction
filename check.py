import pandas as pd
import os

# 読み込み
df_all = pd.read_csv("./all_buildings_for_graph.csv")
df_test = pd.read_csv("./split_lists/test_images.csv")

# 前処理
test_ids = df_test["image_id"].str.lower().tolist()
if "image_id" in test_ids:
    test_ids.remove("image_id")

# image_id抽出関数
def extract_image_id(path):
    base = os.path.basename(path).replace(".tif", "")
    return "_".join(base.split("_")[:2])

# 例として最初の100件で確認
df_sample = df_all.head(100).copy()
df_sample["image_id"] = df_sample["image_path"].apply(extract_image_id)

# 一致チェック
df_sample["in_test"] = df_sample["image_id"].isin(test_ids)
print(df_sample[["image_path", "image_id", "in_test"]].head(10))
print("\n✅ Match found:", df_sample["in_test"].sum())
