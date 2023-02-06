"""
Data loading or dumping.
"""
import os
import json
import logging

def build_key2valueList_dict(d, k, v):
	# does d transfer by address? Yes
	if k not in d:
		d[k] = [v]
	else:
		d[k].append(v)


def build_counter_dict(d, k, count=1):
	# does d transfer by address?
	if k not in d:
		d[k] = count
	else:
		d[k] += count

def load_data_by_whole(input_file):
    """
    A base function for loading json data from a file.
    The whole content in this file is a json type.
    """
    with open(input_file) as reader:
        data = json.load(reader)
    return data


def load_data_by_line(data_file):
    """
    One line in this file is an instance, a json object.
    """
    data = []
    with open(data_file) as reader:
        for line in reader.readlines():
            data.append(json.loads(line))
    print(f"Load {len(data)} instances from {data_file}")
    return data


def load_data_by_block(data_file, lines_of_block):
    """
    Multiple lines in this file is an instance, a json object.

    We can update this function into a auto json object search function by matching a pair of brackets
    """
    data = []
    with open(data_file) as reader:
        counter = 0
        content = ""
        for line in reader.readlines():
            content += line.strip()
            counter += 1
            if counter == lines_of_block:
                data.append(json.loads(content))
                counter = 0
                content = ""
    print(f"Load {len(data)} instances from {data_file}")
    return data


def dump_data_by_line(data, path_to_file):
    with open(path_to_file, 'w') as writer:
        for inst in data:
            writer.write(json.dumps(inst) + '\n')

def dump_data_by_whole(data, path_to_file):
    with open(path_to_file, 'w') as writer:
        json.dump(data, writer, indent=4)

def load_triplet_data(input_file):
    data = []
    triplet_num = 0
    with open(input_file) as reader:
        for line in reader.readlines():
            triplet_num += 1
            insts = json.loads(line)['triplets']
            data += insts
    print(f"Load {len(data)} from {triplet_num} triplets in RelationPrompt File: {input_file}")
    return data

def dump_triplets(triplets, path_to_file):
	with open(path_to_file, "w") as writer:
		for triplet, insts in triplets.items():
			one_line = {"triplets": insts}
			writer.write(json.dumps(one_line) + '\n')


class RelationPromptDataOperator():
    def load_data(self, input_file):
        data = []
        with open(input_file) as reader:
            for line in reader.readlines():
                insts = json.loads(line)['triplets']
                data += insts
        logging.info(f"Load {len(data)} from RelationPrompt File: {input_file}")
        return data

    def dump_triplets(self, triplets, path_file):
        with open(path_file, "w") as writer:
            for triplet, insts in triplets.items():
                one_line = {"triplets": insts}
                writer.write(json.dumps(one_line) + '\n')


    def dump_bagN_triplets(self, triplets, path_file):
        with open(path_file, "w") as writer:
            for triplet, insts in triplets.items():
                if len(insts) > 1:
                    one_line = {"triplets": insts}
                    writer.write(json.dumps(one_line) + '\n')


    def dump_NonNA_bagN_triplets(self, triplets, path_file):
        with open(path_file, "w") as writer:
            for triplet, insts in triplets.items():
                relation = insts[0]['label']
                if len(insts) > 1 and relation != "NA":
                    one_line = {"triplets": insts}
                    writer.write(json.dumps(one_line) + '\n')


    def dump_bag1_triplets(self, triplets, path_file):
        with open(path_file, "w") as writer:
            for triplet, insts in triplets.items():
                if len(insts) == 1:
                    one_line = {"triplets": insts}
                    writer.write(json.dumps(one_line) + '\n')
                

    def dump_NonNA_bag1_triplets(self, triplets, path_file):
        with open(path_file, "w") as writer:
            for triplet, insts in triplets.items():
                relation = insts[0]['label']
                if len(insts) == 1 and relation != "NA":
                    one_line = {"triplets": insts}
                    writer.write(json.dumps(one_line) + '\n')


class OpenNREDataOperator():
    def dump_data(self, data, path_to_file):
        with open(path_to_file, 'w') as writer:
            for inst in data:
                writer.write(json.dumps(inst) + '\n')





def load_data_from_dataset(data_dir, dataset_name):
    """
    Our structure of benchmark is fixed
    To load all sub data in this dataset.
    """
    data_list = []
    for name in ["train", "val", "test"]:
        path_to_file = os.path.join(data_dir, f"{dataset_name}_{name}.txt")
        data = load_data_by_line(path_to_file)
        data_list.append(data)
    rel_file = os.path.join(data_dir, f"{dataset_name}_rel2id.json")
    rel_data = load_data_by_whole(rel_file)
    train_data, val_data, test_data = data_list
    return (train_data, val_data, test_data, rel_data)