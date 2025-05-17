import subprocess

# === script ===
scripts = [
    "train_metadata.py",
    "train_val_split_by_type.py",
    "resnet_by_type.py",
    "test_by_buildingtype.py"
]

for script in scripts:
    print(f"\n🚀 Running {script}")
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Error in {script}:")
        print(result.stderr)