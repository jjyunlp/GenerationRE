"""
2023-02-22, Junjie Yu, Soochow University
For a deep analysis on generated sentences, we randomly output some generated sentences for each relation
in a annotated-friendly manner.
"""

import os
import json
import random
import sys
from typing import List
print(sys.path)
sys.path.insert(0, "/home/jjyu/GenerationForRE" )
from REUtils.data_transform import build_key2valueList_dict
from REUtils.data_io import load_data_from_file
from load_dataset import LoadDataset


class LoadDatasetWithNASplit(LoadDataset):
    """Load and split data into NonNA and NA sets"""
    def __init__(self, base_data_dir, dataset_name, NA_names=None):
        super().__init__(base_data_dir, dataset_name)
        if NA_names is None:
            self.NA_names=["no_relation", "NA", "Other"]
        else:
            self.NA_names = NA_names
        print(self.NA_names)

    def load_training_NA_and_NonNA_data(self,):
        """return (NA data, NonNA data)"""
        data = self.load_training_data()
        NA_data, NonNA_data = self.extract_NA_data(data, self.NA_names)
        return (NA_data, NonNA_data)

    def load_validation_NA_and_NonNA_data(self,):
        data = self.load_validation_data()
        NA_data, NonNA_data = self.extract_NA_data(data, self.NA_names)
        return (NA_data, NonNA_data)

    def load_test_NA_and_NonNA_data(self,):
        pass

    def extract_NA_data(self, data, NA_names):
        """
        NA_names is a list of possible NA tags
        """
        NA_data = []
        NonNA_data = []
        print(len(data))
        for inst in data:
            if inst["relation"] in NA_names:
                NA_data.append(inst)
            else:
                NonNA_data.append(inst)
        if len(NA_data) == 0:
            print(f"No NA data found! Is Na_names={NA_names} correct?")
        return (NA_data, NonNA_data)


class BuildSemiSupervisedScenario():
    """Input a data, and then choose which approach to split into labeled and unlabeled data sets.
    """
    def __init__(self, data):
        print(f"load data {len(data)} instances.")
        self.data = data
    
    def split_data_by_triplets(self, labeled_proportion=0.5):
        """
        Split the data into labeled and unlabeled triplets, with its sentences.
        """
        triplet2insts = self.extract_triplet(self.data)
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

    @classmethod
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


class REDataTransform():
    """transform the generated RE instances"""
    def __init__(self, data) -> None:
        self.data = data
    
    def build_relation_based_data(self):
        relation2insts = {}
        for item in self.data:
            insts = item['triplets']
            for inst in insts:
                label = inst['label']
                build_key2valueList_dict(relation2insts, label, inst)
        print(f"Load {len(relation2insts)} triplets.")
        return relation2insts


class DataOutputForAnnotation():
    """output data in a annotation-friendly manner."""
    def __init__(self, path_to_output) -> None:
        self.path_to_output = path_to_output
    
    def convert_RE_format(self, inst):
        """transfer a RE instance into an annotation-friendly string.
        If the instance is from RelationPrompt generated data, its format:
        {tokens: [w0, w1, w2, ..., wn], head: [w1, w2], tail: [w7], label: str}.
        Transfer to {Sentence: wrapped_sen, Head: head_name, Tail: tail_name, Label: label, IsTrue: 0/1}
        """
        output = {}
        head_ids = inst['head']
        tail_ids = inst['tail']
        head_name = []
        tail_name = []
        wrapped_sen = ""
        for i, tok in enumerate(inst['tokens']):
            # add end marker first, to avoid '[E2] [/E1]' as it should be '[/E1] [E2]' for an adjacent entity situation
            if i == head_ids[-1] + 1:
                wrapped_sen += "[/E1] "
            if i == tail_ids[-1] + 1:
                wrapped_sen += "[/E2] "
            if i == head_ids[0]:
                wrapped_sen += "[E1] "
            if i == tail_ids[0]:
                wrapped_sen += "[E2] "
            wrapped_sen += f"{tok} "
            if i in head_ids:
                head_name.append(tok)
            if i in tail_ids:
                tail_name.append(tok)
        output['Sentence'] = wrapped_sen
        # Then head tail info
        output['Head'] = ' '.join(head_name)
        output['Tail'] = ' '.join(tail_name)
        # label
        output['Label'] = inst['label']
        output['IsTrue'] = 2    # default 2 means unlabel
        output['Error'] = ""

        return output
    
    def write_key_value_data(self, data, N=10)->None:
        """
        data: a dictionary contains a key with its instances
        N: select N instances for each relation.
        """
        with open(self.path_to_output, 'w') as writer:
            for k, insts in data.items():
                writer.write(f"Label is: {k}, Generated Sentence {len(insts)}\n")
                selected_insts = random.sample(insts, k=N)
                for inst in selected_insts:
                    readable_inst = self.convert_RE_format(inst)
                    writer.write(json.dumps(readable_inst, indent=4))
                    writer.write('\n')


if __name__ == "__main__":
    data_dir = "/data/jjyu/RE_Sentence_Generation/Fine_tuned_model/re-tacred_1_9/scale1.0_seed0/gpt2-large/triplet/generator/output_data"
    data_filename = "synthetic_sentence_for_untrain_triplet.json"
    output_filename = "readable_synthetic_sentence_for_annotation.txt"
    path_to_input_file = os.path.join(data_dir, data_filename)
    path_to_output_file = os.path.join(data_dir, output_filename)
    data = load_data_from_file(path_to_input_file)
    data_transform = REDataTransform(data)
    rel2insts = data_transform.build_relation_based_data()
    outputer = DataOutputForAnnotation(path_to_output_file)
    outputer.write_key_value_data(rel2insts, N=10)




    