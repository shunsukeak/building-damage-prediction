import subprocess

scripts = [
    "../predism+GAT+focal_loss/test_1_building_graphs_test_metadata.py",
    "../predism+GAT+focal_loss/test_2_building_graphs_test_metadata_add_shape.py",
    "../predism+GAT+focal_loss/test_3_building_graphs_build_test_graph_per_type.py",
    "../predism+GAT+focal_loss/test_4_building_graphs_test_on_gat.py"
]

for script in scripts:
    print(f"\n🚀 Running {script}")
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Error in {script}:\n{result.stderr}")
