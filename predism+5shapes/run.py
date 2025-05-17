import subprocess

scripts = [
    "1train_metadata_with_shape.py",
    "2adding_with_shape.py",
    "3train_val_split_by_type_with_shape.py",
    "4resnet_by_type_with_shape.py",
    "5test_by_buildingtype_with_shape.py"
]

for script in scripts:
    print(f"\n🚀 Running {script}")
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Error in {script}:")
        print(result.stderr)
