"""
Convert generated data (in RP format) into RE format.
Keep the data distribution in original dataset.
Merge human-annotated, NA data.
"""
"""
将小数据的训练集合，生成的伪数据，NA等合并，构造成新的一套RE语料
"""

import os
import json
import argparse
import logging
from re import T
from typing import List
from pydantic.main import BaseModel

from REUtils.data_transform import load_triplet_data, load_data_by_line, dump_data_by_line
from REUtils.data_transform import build_key2valueList_dict

# 这个代码目前没用了，看看那能不能整合下，data_format.py
class FormatConverter(object):
    """
    RelationPrompt and OpenNRE use different formats of data.
    Thus, we need to convert the data format between two different formats.
    """
    def get_entity_locations(self, c_start, c_end, tokens, ent_str):
        current_size = 0
        entity_index = []
        for i, tok in enumerate(tokens):
            if c_start <= current_size < c_end:
                entity_index.append(i)
            current_size += len(tok)
            if i != len(tokens) - 1:	# not last token
                current_size += 1	# add a space
        
        entity_tokens = " ".join([tokens[x] for x in entity_index])
        # assert head_tokens.lower() == head_name.lower()
        # In NYT10m, some instances are labeled with a correct entity name but error positions.
        # Like, entity="new york", but it will find the "new yorkers" if it is before the real entity "new york"
        # It is a substring problem, very low rate, it is ok.
        unmatched = False
        entity_tokens = entity_tokens.lower()
        ent_str = ent_str.lower()
        if entity_tokens != ent_str:
            if ent_str in entity_tokens or entity_tokens in ent_str:
                print(entity_tokens, ent_str)
            else:
                print(tokens)
                print(entity_tokens, ent_str)
            unmatched = True
        return unmatched, entity_index

    def convert(self, filename):
        """
        这个是OpenNRE data to RelationPrompt data
        """
        data = []
        with open(filename) as reader:
            for line in reader.readlines():
                data.append(json.loads(line.strip()))
        print(f"Load {len(data)} instances from {filename}.")
        triplets = {}
        for inst in data:
            # print(inst)
            tokens = inst['text'].split()
            head_start, head_end = inst['h']['pos']
            head_name = inst['h']['name']
            head_unmatched, head_index = self.get_entity_locations(head_start, head_end, tokens, head_name)

            tail_start, tail_end = inst['t']['pos']
            tail_name = inst['t']['name']
            tail_unmatched, tail_index = self.get_entity_locations(tail_start, tail_end, tokens, tail_name)
            if head_unmatched or tail_unmatched:
                # Skip the sentence with wrong entity position information in the original data
                continue	
            label = inst['relation']
            inst = {
                "tokens": tokens,
                "head": head_index,
                "tail": tail_index,
                "label": label
            }
            triplet = f"{head_name}_{tail_name}_{label}"
            if triplet not in triplets:
                triplets[triplet] = [inst]
            else:
                triplets[triplet].append(inst)
        return triplets

    def get_character_position_info_from_relationprompt_inst(self, tokens: List[str], entity_locations: List[int]):
        """
        input a character-based position
        """
        sen = ""
        name = ""
        pos_start = -1
        pos_end = -1
        for i, token in enumerate(tokens):
            if i in entity_locations:
                name += f"{token} "	# with space
                if i == entity_locations[0]:
                    # entity start here, 
                    pos_start = len(sen)
            sen += f"{token} "	# add token with a space
        pos_end = pos_start + len(name) - 1		# minus space
        return sen.rstrip(), [pos_start, pos_end], name[:-1]

    def get_opennre_position_from_relationprompt_inst(self, entity_locations: List[int]):
        """
        into a token-based position
        Input: list all postion of tokens for an entity
        Output: list start and end token postion of an entity
        the end position should be 1 plus the postion(index) of last token for an entity

        """
        pos_start = entity_locations[0]
        pos_end = entity_locations[-1] + 1
        return [pos_start, pos_end]

    def convert_relationprompt_into_temp(self, data):
        """
        convert relation prompt into opennre.
        sen + character-based position
        """
        opennre_data = []
        for inst in data:
            # construct a new ds format inst
            opennre_inst = {}
            text, h_pos, h_name = self.get_opennre_info_from_relation_prompt_inst(inst['tokens'], inst['head'])
            text, t_pos, t_name = self.get_opennre_info_from_relation_prompt_inst(inst['tokens'], inst['tail'])
            opennre_inst = {
                'text': text,
                'h': {'pos': h_pos, 'name': h_name},
                't': {'pos': t_pos, 'name': t_name},
                'relation': inst['label']
            }
            opennre_data.append(opennre_inst)
        return opennre_data

    def convert_relationprompt_into_opennre(self, data):
        """
        convert relation prompt into opennre.
        token + token-based position e.g., [0, 1] to represent the first token is entity
        """
        opennre_data = []
        for inst in data:
            # construct a new ds format inst
            opennre_inst = {}
            h_pos = self.get_opennre_position_from_relationprompt_inst(inst['head'])
            t_pos = self.get_opennre_position_from_relationprompt_inst(inst['tail'])
            token = inst['tokens']
            if 'head_name' not in inst or 'tail_name' not in inst:
                head_name = " ".join(token[h_pos[0]:h_pos[1]])
                tail_name = " ".join(token[t_pos[0]:t_pos[1]])
            else:
                head_name = inst['head_name']
                tail_name = inst['tail_name']

            opennre_inst = {
                'token': token,
                'h': {'pos': h_pos, 'name': head_name},
                't': {'pos': t_pos, 'name': tail_name},
                'relation': inst['label']
            }
            opennre_data.append(opennre_inst)
        return opennre_data

    def convert_relationprompt_into_opennre_with_entity_check(self, data, entityName2Id):
        """
        entityName2Id is used to check if the generated entity is valid
        有些情况，RelationPrompt会生成一些substring的entity问题
        """
        ds_data = []
        for inst in data:
            # construct a new ds format inst
            ds_inst = {}
            text, h_pos, h_name = self.get_opennre_info_from_relation_prompt_inst(inst['tokens'], inst['head'])
            text, t_pos, t_name = self.get_opennre_info_from_relation_prompt_inst(inst['tokens'], inst['tail'])
            if h_name not in entityName2Id:
                print(f"{h_name}: can not found in entity id")
                print(inst)
                print("\n")
                continue
            if t_name not in entityName2Id:
                print(f"{t_name}: can not found in entity id")
                print(inst)
                print("\n")
                continue
            h_id = entityName2Id[h_name]
            t_id = entityName2Id[t_name]
            ds_inst = {
                'text': text,
                'h': {'pos': h_pos, 'name': h_name, 'id': h_id},
                't': {'pos': t_pos, 'name': t_name, 'id': t_id},
                'relation': inst['label']
            }
            ds_data.append(ds_inst)
        return ds_data


class RPdataIntoREData():
    """
    RelationPrompt and OpenNRE use different formats of data.
    Thus, we need to convert the data format between two different formats.
    RP data is triplets based json type.
    RE data is one instance one json type
    主要不同：
        1. 一些columns上的不同
        2. entity position的不同
    RP data
        a list of instances, no "triplets"
        已经是处理过的了。
        tokens
        head: a continuous sequence of index to point the index of tokens in an entity
        tail: same with head
        label: str relation
    RE data:
        token
        h : {pos: [strat, end], name: str}
        t : {pos: [strat, end], name: str}
        relation: str
    """
    def get_entity_locations(self, c_start, c_end, tokens, ent_str):
        current_size = 0
        entity_index = []
        for i, tok in enumerate(tokens):
            if c_start <= current_size < c_end:
                entity_index.append(i)
            current_size += len(tok)
            if i != len(tokens) - 1:	# not last token
                current_size += 1	# add a space
        
        entity_tokens = " ".join([tokens[x] for x in entity_index])
        # assert head_tokens.lower() == head_name.lower()
        # In NYT10m, some instances are labeled with a correct entity name but error positions.
        # Like, entity="new york", but it will find the "new yorkers" if it is before the real entity "new york"
        # It is a substring problem, very low rate, it is ok.
        unmatched = False
        entity_tokens = entity_tokens.lower()
        ent_str = ent_str.lower()
        if entity_tokens != ent_str:
            if ent_str in entity_tokens or entity_tokens in ent_str:
                print(entity_tokens, ent_str)
            else:
                print(tokens)
                print(entity_tokens, ent_str)
            unmatched = True
        return unmatched, entity_index

    def convert(self, filename):
        """
        这个是OpenNRE data to RelationPrompt data
        """
        data = []
        with open(filename) as reader:
            for line in reader.readlines():
                data.append(json.loads(line.strip()))
        print(f"Load {len(data)} instances from {filename}.")
        triplets = {}
        for inst in data:
            # print(inst)
            tokens = inst['text'].split()
            head_start, head_end = inst['h']['pos']
            head_name = inst['h']['name']
            head_unmatched, head_index = self.get_entity_locations(head_start, head_end, tokens, head_name)

            tail_start, tail_end = inst['t']['pos']
            tail_name = inst['t']['name']
            tail_unmatched, tail_index = self.get_entity_locations(tail_start, tail_end, tokens, tail_name)
            if head_unmatched or tail_unmatched:
                # Skip the sentence with wrong entity position information in the original data
                continue	
            label = inst['relation']
            inst = {
                "tokens": tokens,
                "head": head_index,
                "tail": tail_index,
                "label": label
            }
            triplet = f"{head_name}_{tail_name}_{label}"
            if triplet not in triplets:
                triplets[triplet] = [inst]
            else:
                triplets[triplet].append(inst)
        return triplets

    def get_character_position_info_from_relationprompt_inst(self, tokens: List[str], entity_locations: List[int]):
        """
        input a character-based position
        """
        sen = ""
        name = ""
        pos_start = -1
        pos_end = -1
        for i, token in enumerate(tokens):
            if i in entity_locations:
                name += f"{token} "	# with space
                if i == entity_locations[0]:
                    # entity start here, 
                    pos_start = len(sen)
            sen += f"{token} "	# add token with a space
        pos_end = pos_start + len(name) - 1		# minus space
        return sen.rstrip(), [pos_start, pos_end], name[:-1]

    def get_opennre_position_from_relationprompt_inst(self, entity_locations: List[int]):
        """
        into a token-based position
        Input: list all postions (indexes) of tokens for an entity
        Output: list start and end token postion (+1) of an entity
            the end position should be 1 plus the postion(index) of last token for an entity
            E.g, "Yao Ming is born in ShangHai", the position for entity "Yao Ming" is [0, 2]

        """
        pos_start = entity_locations[0]
        pos_end = entity_locations[-1] + 1
        return [pos_start, pos_end]

    def convert_relationprompt_into_temp(self, data):
        """
        convert relation prompt into opennre.
        sen + character-based position
        """
        opennre_data = []
        for inst in data:
            # construct a new ds format inst
            opennre_inst = {}
            text, h_pos, h_name = self.get_opennre_info_from_relation_prompt_inst(inst['tokens'], inst['head'])
            text, t_pos, t_name = self.get_opennre_info_from_relation_prompt_inst(inst['tokens'], inst['tail'])
            opennre_inst = {
                'text': text,
                'h': {'pos': h_pos, 'name': h_name},
                't': {'pos': t_pos, 'name': t_name},
                'relation': inst['label']
            }
            opennre_data.append(opennre_inst)
        return opennre_data

    def convert_relationprompt_into_opennre(self, data):
        """
        convert relation prompt into opennre.
        token + token-based position e.g., [0, 1] to represent the first token is entity
        Input: 
            List[RP_inst]
        Output:
            List[RE_inst]
        """
        opennre_data = []
        for inst in data:
            # construct a new ds format inst
            opennre_inst = {}
            h_pos = self.get_opennre_position_from_relationprompt_inst(inst['head'])
            t_pos = self.get_opennre_position_from_relationprompt_inst(inst['tail'])
            token = inst['tokens']
            if 'head_name' not in inst or 'tail_name' not in inst:
                head_name = " ".join(token[h_pos[0]:h_pos[1]])
                tail_name = " ".join(token[t_pos[0]:t_pos[1]])
            else:
                head_name = inst['head_name']
                tail_name = inst['tail_name']

            opennre_inst = {
                'token': token,
                'h': {'pos': h_pos, 'name': head_name},
                't': {'pos': t_pos, 'name': tail_name},
                'relation': inst['label']
            }
            opennre_data.append(opennre_inst)
        return opennre_data

    def build_triplet2insts(self, data, sep="#-#"):
        """build a dict that triplet as key, list[inst] as value
        triplet = {head}{sep}{tail}{sep}{rel}
        Input:
            List[inst]
        Output:
            dict
        """
        triplet2insts = {}
        for inst in data:
            h_name = inst['h']['name']
            t_name = inst['t']['name']
            rel = inst['relation']
            triplet = sep.join([h_name, t_name, rel])
            build_key2valueList_dict(triplet2insts, k=triplet, v=inst)
        return triplet2insts

    def convert_relationprompt_into_opennre_with_entity_check(self, data, entityName2Id):
        """
        entityName2Id is used to check if the generated entity is valid
        有些情况，RelationPrompt会生成一些substring的entity问题
        但现在优化了代码，以token为单位进行匹配了，所以，应该不存在这个问题了。对于RE-TACRED这个，我们也没有entityName2Id
        """
        ds_data = []
        for inst in data:
            # although data is triplets based formation,
            # construct a new ds format inst
            ds_inst = {}
            text, h_pos, h_name = self.get_opennre_info_from_relation_prompt_inst(inst['tokens'], inst['head'])
            text, t_pos, t_name = self.get_opennre_info_from_relation_prompt_inst(inst['tokens'], inst['tail'])
            if h_name not in entityName2Id:
                print(f"{h_name}: can not found in entity id")
                print(inst)
                print("\n")
                continue
            if t_name not in entityName2Id:
                print(f"{t_name}: can not found in entity id")
                print(inst)
                print("\n")
                continue
            h_id = entityName2Id[h_name]
            t_id = entityName2Id[t_name]
            ds_inst = {
                'text': text,
                'h': {'pos': h_pos, 'name': h_name, 'id': h_id},
                't': {'pos': t_pos, 'name': t_name, 'id': t_id},
                'relation': inst['label']
            }
            ds_data.append(ds_inst)
        return ds_data


class CollectGeneratedData():
    """Collect instances for untrain triplets
    """
    def __init__(self, triplet2insts, triplet2num):
        """
        Input:
            triplet2insts: generated insts
            triplet2num: required num for each triplet
            triplet2insts < triplet2num: as model may generate 0 sentence for a triplet
        """
        self.triplet2insts = triplet2insts
        self.triplet2num = triplet2num
    
    def fill_sentences_for_triplet(self,):
        """If no enough sentences, just skip.
        I think, we should do some analysis. For example, rel_A needs 100, but only 40"""
        output_data = []
        for triplet, insts in self.triplet2insts.items():
            num = self.triplet2num[triplet]
            if len(insts) > num:
                output_data += insts
            else:
                output_data += insts[:num]
        return output_data


class TripletDataset(BaseModel):
    """
    The data structure for generation: triplet -> sentence
    """
    triplet2num: dict

    @classmethod
    def convert(cls, line):
        """convert untrain triplet input into HeadTailRelNum"""
        inst = json.loads(line)
        triplet = inst["triplet"]
        if "insts" in inst:
            num = len(inst['insts'])
        else:
            num = 0
        # return {"head": head, "tail": tail, "rel": rel, "num": num}
        return [triplet, num]
        return [head, tail, rel, num]

    @classmethod
    def load(cls, path: str, num: int):
        """num is the default number of required sentences for each triplet to generate"""
        triplet
        with open(path) as f:
            for line in f:
                triplet, num = cls.convert(line)

            head_tail_rel_num = [cls.convert(line) for line in f]
        return cls(head_tail_rel_num=head_tail_rel_num)

    def get_triplet2num(self, sep="#-#") -> dict:
        triplet2num = {}
        for head, tail, rel, num in self.head_tail_rel_num:
            triplet = sep.join([head, tail, rel])
            triplet2num[triplet] = num
        return triplet2num

def load_triplet2num(input_file):
    triplet2num = {}
    with open(input_file) as f:
        for line in f:
            inst = json.loads(line)
            triplet = inst["triplet"]
            if "insts" in inst:
                num = len(inst['insts'])
            else:
                num = 0
            triplet2num[triplet] = num
    return triplet2num



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Supervised RE data by human-annotated data and generated data.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--scale", type=str, default="1.0", required=True, help="1.0 means use all training data. Others are 0.1, 0.2, 0.5")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt", type=str, default="template", choices=["triplet", "template", "qa"])
    args = parser.parse_args()

    # We obtain a generated data, a special format.
    # Collect instances into triplet2[insts], the inst is the RE format
    # Obtain triplet2num, the number of sentences for each triplet in original training set.
    # collect sentences under the number constraint. If exists more, then cut, else (1) ignore, (2) add templates

    base_dir = "/data/jjyu/RE_Sentence_Generation/Fine_tuned_model/"
    generated_datadir = os.path.join(base_dir, args.dataset, f"scale{args.scale}_seed{args.seed}", args.prompt, "generator", "output_data")
    generated_data_file = os.path.join(generated_datadir, "synthetic_sentence_for_untrain_triplet.json")
    generated_data = load_triplet_data(generated_data_file)
    print(f"generated data size before convert: {len(generated_data)}")
    # convert RP data into RE data for generated dataset.
    convertor = RPdataIntoREData()
    generated_data = convertor.convert_relationprompt_into_opennre(generated_data)
    print(f"generated data size: {len(generated_data)}")
    generated_triplet2insts = convertor.build_triplet2insts(generated_data, sep="#-#")

    data_dir = "/home/jjyu/GenerationForRE/Benchmark"
    untrain_triplet_file = os.path.join(data_dir, args.dataset, "untrain_triplet.txt")
    untrain_triplet2num= load_triplet2num(untrain_triplet_file)
    collector = CollectGeneratedData(generated_triplet2insts, untrain_triplet2num)
    collected_generated_data = collector.fill_sentences_for_triplet()
    print(collected_generated_data)
    exit()

    RE_datadir = f"../benchmark/{args.dataset}_RE/"     # the small dataset comes from the original human-annotated data
    RE_dataset = f"{args.dataset}_RE"


    RP_datadir = f"../benchmark/{args.dataset}_RP/"     # convert the small dataset into RelationPrompt formation
    RP_dataset = f"{args.dataset}_RP"

    target_datadir = f"../benchmark/{args.dataset}_Generated"            # the final dataset for train the RE model
    target_dataset = f"{args.dataset}_Generated"     # the final dataset with generated sentences
    if not os.path.exists(RP_datadir):
        print(f"Input dataset is not exist: {RP_datadir}")
        exit()
    if not os.path.exists(RE_datadir):
        print(f"Input dataset is not exist: {RE_datadir}")
        exit()
    if not os.path.exists(target_datadir):
        print(f"Create output dataset folder: {target_datadir}")
        os.mkdir(target_datadir)

    human_train_data_file = os.path.join(RE_datadir, f"{RE_dataset}_train.txt")
    human_data = load_data_by_line(human_train_data_file)
    print(f"small train data size: {len(human_data)}")

    human_NA_data_file = os.path.join(RE_datadir, f"{RE_dataset}_train_NA.txt")
    human_NA_data = load_data_by_line(human_NA_data_file)
    print(f"train NA data size: {len(human_NA_data)}")


    train_data = human_data + human_NA_data + generated_data
    train_data_file = os.path.join(target_datadir, f"{target_dataset}_train.txt")
    dump_data_by_line(train_data, train_data_file)

    # original val set, test set, rel2id_file
    base_dataset = args.dataset.split("_")[0]
    if base_dataset not in ['re-tacred', 'semeval']:
        print(f"Error base dataset: {base_dataset} from input dataset: {args.dataset}")
        exit()
    base_datadir = f"../benchmark/{base_dataset}"
    copy_val = f"cp {base_datadir}/{base_dataset}_val.txt {target_datadir}/{target_dataset}_val.txt"
    os.system(copy_val)
    copy_test = f"cp {base_datadir}/{base_dataset}_test.txt {target_datadir}/{target_dataset}_test.txt"
    os.system(copy_test)
    copy_rel2id = f"cp {base_datadir}/{base_dataset}_rel2id.json {target_datadir}/{target_dataset}_rel2id.json"
    os.system(copy_rel2id)


