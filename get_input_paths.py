import os
import json
import glob

paths = []

root_dir = "/Users/jaihoonkim/Desktop/wkst/codes/warp/omniobject/assets"
total_files = 0

for cat_dir in glob.glob(f"{root_dir}/*"):
    for mesh_dir in glob.glob(f"{cat_dir}/*"):
        mesh_path = os.path.join(mesh_dir, "Scan")
        paths.append(mesh_path)

print("TOtal files", len(paths))

with open(f"input_models_path.json", "w") as f:
    json.dump(paths, f, indent=2)
