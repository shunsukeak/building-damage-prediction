import subprocess

scripts = [
    "1train_metadata.py",
    "2train_val_split_by_type.py",
    "3resnet_by_type.py",
    "4test_by_buildingtype.py"
]

for script in scripts:
    print(f"\n🚀 Running {script}")
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Error in {script}:")
        print(result.stderr)