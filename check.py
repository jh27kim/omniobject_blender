import json
import os

with open("./input_models_path.json", "r") as f:
    metadata = json.load(f)

saved_obj = os.listdir("./views")
for i in range(len(metadata)):
    oid = metadata[i].split("/")[-2]
    if oid not in saved_obj:
        print(oid)
