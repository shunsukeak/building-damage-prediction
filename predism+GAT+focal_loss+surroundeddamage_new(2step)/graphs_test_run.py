import subprocess

scripts = [
    "test_1_building_graphs_test_metadata.py",
    "test_2_building_graphs_test_metadata_add_shape.py",
    "test_3_building_graphs_build_test_graph_per_type.py",
    "test_4_building_graphs_test_on_gat.py"
]

for script in scripts:
    print(f"\n🚀 Running {script}")
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Error in {script}:\n{result.stderr}")


# import subprocess
# from pathlib import Path

# current_dir = Path(__file__).parent

# scripts = [
#     "test_1_building_graphs_test_metadata.py",
#     "test_2_building_graphs_test_metadata_add_shape.py",
#     "test_3_step1.py",
#     "test_4_step1.py",
#     "test_3_step2.py",
#     "test_4_step2.py",
# ]

# for script in scripts:
#     script_path = current_dir / script
#     print(f"\n🚀 Running {script_path}")
#     result = subprocess.run(["python3", str(script_path)], capture_output=True, text=True)
#     print(result.stdout)
#     if result.stderr:
#         print(f"⚠️ Error in {script}:\n{result.stderr}")