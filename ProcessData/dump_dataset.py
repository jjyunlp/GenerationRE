import os
from REUtils.data_io import dump_data_to_file, dump_labels_to_file

class DumpDataset():
    """a base class for dumping datasets"""
    def __init__(self, base_data_dir, dataset_name):
        print(base_data_dir)
        print(dataset_name)
        self.path_to_train = os.path.join(base_data_dir, dataset_name, 
                                          f"{dataset_name}_train.txt")
        self.path_to_val = os.path.join(base_data_dir, dataset_name, 
                                        f"{dataset_name}_val.txt")
        self.path_to_test = os.path.join(base_data_dir, dataset_name, 
                                         f"{dataset_name}_test.txt")
        self.path_to_label = os.path.join(base_data_dir, dataset_name, 
                                         f"{dataset_name}_rel2id.json")
        
    def check_file_exists(self, path_to_file):
        if os.path.exists(path_to_file):
            return True

    def dump_training_data(self, data, overwrite=False):
        dump_data_to_file(self.path_to_train, data, overwrite)
        print(f"Dumped {len(data)} instances to file: {self.path_to_train}")
    
    def dump_validation_data(self, data, overwrite=False):
        dump_data_to_file(self.path_to_val, data, overwrite)
        print(f"Dumped {len(data)} instances to file: {self.path_to_val}")
        return data

    def dump_test_data(self, data, overwrite=False):
        dump_data_to_file(self.path_to_test, data, overwrite)
        print(f"Dumped {len(data)} instances to file: {self.path_to_test}")
        return data

    def dump_label_mapping(self, data, overwrite=False):
        dump_labels_to_file(data, self.path_to_label, overwrite)
        print(f"label2id file is saved at {self.path_to_label}")