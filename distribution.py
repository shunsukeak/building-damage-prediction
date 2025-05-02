import pandas as pd

# CSVファイルのパス
csv_path = "./cropped_building_metadata.csv"

# CSVを読み込み
df = pd.read_csv(csv_path)

# ラベル列の確認（カラム名が違う場合はここを修正）
label_column = 'label'

# クラス分布を集計
class_counts = df[label_column].value_counts().sort_index()
total = class_counts.sum()

print("✅ Class Distribution from CSV")
for cls, count in class_counts.items():
    percent = count / total * 100
    print(f"Class {cls}: {count} ({percent:.2f}%)")