df_all = pd.read_csv("./all_buildings_for_graph.csv")
df_test = pd.read_csv("./split_lists/test_images.csv")
test_ids = df_test["image_id"].dropna().str.lower().tolist()
test_ids = [tid for tid in test_ids if tid != "image_id"]

def extract_image_id(path):
    filename = os.path.basename(path).replace(".tif", "")
    return "_".join(filename.split("_")[:2]).lower()

df_sample = df_all.head(1000).copy()
df_sample["image_id"] = df_sample["image_path"].apply(extract_image_id)
df_sample["in_test"] = df_sample["image_id"].isin(test_ids)

print(df_sample[["image_path", "image_id", "in_test"]].head(10))
print("\n✅ Match found:", df_sample["in_test"].sum())
