import subprocess

scripts = [
    "test_1_building_graphs_test_metadata.py",
    "test_2_building_graphs_test_metadata_add_shape.py",
    "test_3_building_graphs_build_test_graph_per_type.py",
    "test_4_building_graphs_test_on_gcn.py"
]

for script in scripts:
    print(f"\n🚀 Running {script}")
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Error in {script}:\n{result.stderr}")
