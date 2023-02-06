"""
Convert a supervised dataset into a semisupervised scenario.
Half data are prepared for training, and half data are for generation.
Triplets as keys.
"""

import os
import json
import random
from data_load_and_dump import load_data_from_file
from data_transform import build_key2valueList_dict

class LoadDataset():
    def __init__(self, base_data_dir, dataset_name):
        print(base_data_dir)
        print(dataset_name)
        self.path_to_train = os.path.join(base_data_dir, dataset_name, 
                                          f"{dataset_name}_train.txt")
        self.path_to_val = os.path.join(base_data_dir, dataset_name, 
                                        f"{dataset_name}_val.txt")
        self.path_to_test = os.path.join(base_data_dir, dataset_name, 
                                         f"{dataset_name}_test.txt")
        
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



class BuildSemiSupervisedScenario():
    """Input a data, and then choose which approach to split into labeled and unlabeled data sets.
    """
    def __init__(self, data):
        print(f"load {len(data)} instances")
        NA_data, NonNA_data = self.extract_NA_data(data)
        self.NA_data = NA_data
        self.NonNA_data = NonNA_data
    
    def split_data_by_triplets(self, labeled_proportion=0.5):
        """
        Split the data into labeled and unlabeled triplets, with its sentences.
        """
        triplet2insts = self.extract_triplet(self.NonNA_data)
        triplets = list(triplet2insts.keys())
        test = [x for x in range(10)]
        random.shuffle(test)
        print(test)

        random.shuffle(triplets)
        labeled_num = int(len(triplets) * labeled_proportion)
        labeled_triplets, unlabeled_triplets = triplets[:labeled_num], triplets[labeled_num:]
        print(len(labeled_triplets))
        print(labeled_triplets[:10])
        labeled_triplet2insts = {key: triplet2insts[key] for key in labeled_triplets}
        unlabeled_triplet2insts = {key: triplet2insts[key] for key in unlabeled_triplets}
        return (labeled_triplet2insts, unlabeled_triplet2insts)

    @classmethod
    def extract_NA_data(self, data, NA_names=["no_relation", "NA", "Other"]):
        NA_data = []
        NonNA_data = []
        print(len(data))
        for inst in data:
            if inst["relation"] in NA_names:
                NA_data.append(inst)
            else:
                NonNA_data.append(inst)
        return (NA_data, NonNA_data)

    @classmethod
    def extract_triplet(self, data):
        triplet2insts = {}
        for inst in data:
            head = inst['h']['name']
            tail = inst['t']['name']
            rel = inst['relation']
            triplet_name = f"{head}#-#{tail}#-#{rel}"
            build_key2valueList_dict(triplet2insts, triplet_name, inst)
        print(len(triplet2insts))
        return triplet2insts


class DumpLowResourceDataset():
    """Dump data into a new dictionary for Generation."""
    def __init__(self, base_dir, dataset, scales, seeds) -> None:
        self.base_dir = base_dir    # the directory for save new datasets
        self.dataset = dataset
        self.scales = scales
        self.seeds = seeds

    def dump_semi_dataset(self, labeled_triplet2insts, unlabeled_triplet2insts, val_triplet2insts):
        """Dump data sets into a dictionary, which is the input of NEXT Generation Module.
        Input:
            triplet2insts: a dictionary, key is the labeled triplets, value is its insts in a list
        Output1: A training set.
        Output2: a triplet set with the required number of sentence for each triplet.

        """
        path_to_labeled_triplet = os.path.join(self.base_dir, "train_triplet.txt")
        path_to_unlabeled_triplet = os.path.join(self.base_dir, "untrain_triplet.txt")
        # First, save original labeled triplets and unlabeled triplets
        self.dump_triplet2insts(path_to_labeled_triplet, labeled_triplet2insts)
        self.dump_triplet2insts(path_to_unlabeled_triplet, unlabeled_triplet2insts)

        # Second, save the original labeled_triplet, unlabeled triplet and validation set as flat
        path_to_train_sentence = os.path.join(self.base_dir, "train_sentence.txt")
        self.dump_triplet2insts_as_flat(path_to_train_sentence, labeled_triplet2insts)
        path_to_untrain_sentence = os.path.join(self.base_dir, "untrain_sentence.txt")
        self.dump_triplet2insts_as_flat(path_to_untrain_sentence, unlabeled_triplet2insts)
        
        path_to_val_sentence = os.path.join(self.base_dir, "val_sentence.txt")
        self.dump_triplet2insts_as_flat(path_to_val_sentence, val_triplet2insts)

        # Traverse the seeds before scales, hence, set 1.0 contains set 0.5, set 0.5 contains set 0.2, and so on.
        for seed in self.seeds:
            test = [x for x in range(10)]
            random.seed(seed)
            print(test)
            random.shuffle(test)
            print(test)
            # random shuffle the labeled triplets
            triplets = list(labeled_triplet2insts.keys())
            random.shuffle(triplets)
            for scale in self.scales:
                print(scale)
                actual_num = int(len(triplets) * scale)
                selected_triplets = triplets[:actual_num]
                selected_triplet2insts = {key: labeled_triplet2insts[key] for key in selected_triplets}
                path_to_subset_triplet = os.path.join(self.base_dir, f"train_sentence_scale{scale}_seed{seed}")
                self.dump_triplet2insts_as_flat(path_to_subset_triplet, selected_triplet2insts)


    def dump_triplet2insts_as_flat(self, filename, triplet2insts):
        """
        Add the key (triplet name) into inst's first column, and then dump the insts into file.
        One inst one line.
        """
        with open(filename, 'w') as writer:
            for triplet, insts in triplet2insts.items():
                for inst in insts:
                    new_inst = {"triplet": triplet}
                    new_inst.update(inst)
                    writer.write(json.dumps(new_inst) + '\n')

    def dump_triplet2insts(self, filename, triplet2insts):
        with open(filename, 'w') as writer:
            for triplet, insts in triplet2insts.items():
                new_triplet_object = {"triplet": triplet, "insts": insts}
                writer.write(json.dumps(new_triplet_object) + '\n')








if __name__ == "__main__":
    #loader = LoadDataset("/home/jjyu/GenerationForRE/Datasets", "re-tacred")
    seed = 2
    # This seed is for triplets splitting
    random.seed(seed)
    dataset = "re-tacred"
    loader = LoadDataset("Dataset", "re-tacred")
    print(loader)
    train_data = loader.load_training_data()
    val_data = loader.load_validation_data()    # val data is also needed for training the generator, a simple check of the PPL
    # Directly call the callmethod to get NonNA data of validation
    val_NA_data, val_NonNA_data = BuildSemiSupervisedScenario.extract_NA_data(val_data)
    val_triplet2insts = BuildSemiSupervisedScenario.extract_triplet(val_NonNA_data)

    builder = BuildSemiSupervisedScenario(train_data)
    labeled_triplet2insts, unlabeled_triplet2insts = builder.split_data_by_triplets(labeled_proportion=0.5)

    base_dir = f"/home/jjyu/GenerationForRE/Benchmark/{dataset}_for_generation"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    scales = [0.1, 0.2, 0.5]
    seeds = [0, 1, 2]   # These seeds are used to extract the low-resource training data
    dumper = DumpLowResourceDataset(base_dir, dataset, scales=scales, seeds=seeds)
    dumper.dump_semi_dataset(labeled_triplet2insts, unlabeled_triplet2insts, val_triplet2insts=val_triplet2insts)
    



    