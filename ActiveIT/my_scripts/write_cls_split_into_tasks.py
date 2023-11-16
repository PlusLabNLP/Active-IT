import json
from tqdm import tqdm
import os

def load_file(task_name):
    task_file = os.path.join("../natural-instructions/tasks", task_name)
    return json.load(open(task_file, "r"))
def write_file(task_name, new_data):
    task_file = os.path.join("../natural-instructions/tasks", task_name)
    with open(task_file, "w") as F:
        json.dump(new_data, F, indent = 4)

meta_data_file = "task_metadata.json"
meta_data = json.load(open(meta_data_file, "r"))

print("Writing Split and CLS_GEN tag into task file. We only have this information for EN tasks.")
for task in tqdm(meta_data.keys()):
    if "Split" in meta_data[task].keys():
        Split = meta_data[task]["Split"]
    else:
        Split = None
    if "CLS_GEN" in meta_data[task].keys():
        CLS_GEN = meta_data[task]["CLS_GEN"]
    else:
        CLS_GEN = None
    if (CLS_GEN is not None) or (Split is not None):
        task_data = load_file(task)
        if Split is not None:
            task_data['Split'] = Split
        if CLS_GEN is not None:
            task_data['CLS_GEN'] = CLS_GEN
        write_file(task, task_data)

print("Done...")