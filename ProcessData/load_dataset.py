import os
from REUtils.data_io import load_data_from_file, load_labels_from_file


class LoadDataset():
    """a base class for loading a dataset"""""
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

    def load_training_data(self,):
        if not os.path.exists(self.path_to_train):
            print(f"File {self.path_to_train} not exist! Exit!")
            exit()
        data = load_data_from_file(self.path_to_train)
        print(f"Loading {len(data)} instances from file: {self.path_to_train}")
        return data
    
    def load_validation_data(self,):
        if not os.path.exists(self.path_to_val):
            print(f"File {self.path_to_val} not exist! Exit!")
            exit()
        data = load_data_from_file(self.path_to_val)
        print(f"Loading {len(data)} instances from file: {self.path_to_val}")
        return data

    def load_test_data(self,):
        if not os.path.exists(self.path_to_test):
            print(f"File {self.path_to_val} not exist! Exit!")
            exit()
        data = load_data_from_file(self.path_to_test)
        print(f"Loading {len(data)} instances from file: {self.path_to_test}")
        return data

    def load_label_mapping(self,):
        data = load_labels_from_file(self.path_to_label)
        return data


