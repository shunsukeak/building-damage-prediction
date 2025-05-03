import pandas as pd

# CSVファイルのパス
csv_path = "./cropped_building_metadata.csv"

# CSVを読み込み
df = pd.read_csv(csv_path)

# 災害タイプとラベル列の確認（必要に応じてカラム名を修正）
disaster_column = 'disaster_type'
label_column = 'label'

# 各災害タイプでループ
for disaster_type, group_df in df.groupby(disaster_column):
    print(f"\n🌍 Disaster Type: {disaster_type}")
    class_counts = group_df[label_column].value_counts().sort_index()
    total = class_counts.sum()
    for cls, count in class_counts.items():
        percent = count / total * 100
        print(f"  Class {cls}: {count} ({percent:.2f}%)")
