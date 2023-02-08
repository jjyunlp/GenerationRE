"""
In order to debug quickly, we use 10% of re-tacred to run.
"""

import os
import random
from load_dataset import LoadDataset
from dump_dataset import DumpDataset


dataset = "re-tacred"
output_dataset = "re-tacred_0.1"

loader = LoadDataset("Dataset", dataset)
train_data = loader.load_training_data()
val_data = loader.load_validation_data()
test_data = loader.load_test_data()
rel2id_data = loader.load_label_mapping()

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

train_data = train_data[: int(len(train_data) * 0.1)]
val_data = train_data[: int(len(val_data) * 0.1)]
test_data = train_data[: int(len(test_data) * 0.1)]
output_dir = os.path.join("Dataset", output_dataset)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

base_dir = "Dataset"
dumper = DumpDataset(base_dir, output_dataset)
dumper.dump_training_data(train_data, overwrite=True)
dumper.dump_validation_data(val_data, overwrite=True)
dumper.dump_test_data(test_data, overwrite=True)
print(rel2id_data)
dumper.dump_label_mapping(rel2id_data, overwrite=True)


