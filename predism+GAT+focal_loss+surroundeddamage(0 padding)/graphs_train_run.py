import subprocess

scripts = [
    "train_1_build_graphs_metadata.py",
    "train_2_build_graphs_by_disaster_type.py",
    "train_3_build_graphs_extract_resnet_features_and_merge.py",
    "train_4_build_graphs_train_gat_by_type.py"
]

for script in scripts:
    print(f"\n🚀 Running {script}")
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Error in {script}:\n{result.stderr}")
