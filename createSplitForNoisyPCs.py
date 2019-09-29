import json
import os

# TODO: currently creates test split for noisy CHAIRS --> should be class agnostic

data_source = '/Volumes/warm_blue/deepsdf/data'

with open('examples/splits/sv2_chairs_train.json', "r") as f:
    train_split = json.load(f)
with open('examples/splits/sv2_chairs_test.json', "r") as f:
    test_split = json.load(f)

dataset_name = ''
class_name = ''

# Fetch Dai's point cloud reps
pc_filenames = []
for root, dirs, files in os.walk("/Volumes/warm_blue/datasets/partial_scans/test-images_dim32_sdf_pc/03001627",
                                 topdown=False):
    for name in files:
        if name.endswith('ply'):
            pc_filenames.append(name[:-9])

# Fetch training chairs for filtering neg
train_chairs = []
for dataset in train_split:
    dataset_name = dataset
    for class_name in train_split[dataset]:
        class_name = class_name
        for instance_name in train_split[dataset][class_name]:
            train_chairs.append(instance_name)

# Fetch representation test chairs for filtering positive
test_chairs = []
for dataset in test_split:
    for class_name in test_split[dataset]:
        for instance_name in test_split[dataset][class_name]:
            test_chairs.append(instance_name)

res = [i for i in pc_filenames if i not in train_chairs]
# print(len(res)) --> 80

res2 = [i for i in res if i in test_chairs]
# print(len(res2))  -->18

split_map = {}
dataset_map = {class_name: res2}

split_map[dataset_name] = dataset_map

with open('./examples/splits/sv2_chairs_test_noise.json', "w") as f:
    json.dump(split_map, f, indent=2)
