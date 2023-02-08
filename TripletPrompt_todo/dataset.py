import json
import random
from collections import Counter
from pathlib import Path
from typing import List

from fire import Fire
from pydantic.main import BaseModel
from tqdm import tqdm

from utils import (RelationSentence, WikiDataset, delete_checkpoints,
                   load_wiki_relation_map, mark_fewrel_entity)


class Sentence(BaseModel):
    triplets: List[RelationSentence]

    @property
    def tokens(self) -> List[str]:
        return self.triplets[0].tokens

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def assert_valid(self):
        assert len(self.tokens) > 0
        for t in self.triplets:
            assert t.text == self.text
            assert len(t.head) > 0
            assert len(t.tail) > 0
            assert len(t.label) > 0


class Dataset(BaseModel):
    sents: List[Sentence]

    def get_labels(self) -> List[str]:
        return sorted(set(t.label for s in self.sents for t in s.triplets))
    
    def get_heads_and_tails(self) -> List:
        head_and_tail_list = []
        for s in self.sents:
            head, label, tail= s.triplets[0].as_tuple() # triplets是一个list,包含整个bag
            head_and_tail_list.append([head, tail])
        return head_and_tail_list

    def get_heads_tails_relations(self) -> List:
        # this is a function only work for supervised data that we use triplets from part of the data.
        # The input data is with the same format as training data 
        head_tail_relation_list = []
        for s in self.sents:
            head, label, tail= s.triplets[0].as_tuple() # triplets是一个list,包含整个bag
            num = len(s.triplets)   # how many sentences support this triplet in the original human-annotated data
            head_tail_relation_list.append([head, tail, label, num])
        return head_tail_relation_list
    
    def get_heads_tails_descriptions(self) -> List:
        """return List[head, tail, description, number, rel]"""
        items = []
        for s in self.sents:
            head, tail= s.triplets[0].get_entity_names() # triplets是一个list,包含整个bag
            description = s.triplets[0].description
            rel = s.triplets[0].label
            num = len(s.triplets)   # how many sentences support this triplet in the original human-annotated data
            items.append([head, tail, description, num, rel])
        return items

    def get_heads_tails_questions(self) -> List:
        # this is a function only work for supervised data that we use triplets from part of the data.
        # The input data is with the same format as training data 
        head_tail_relation_list = []
        for s in self.sents:
            head, tail, rel, question = s.triplets[0].get_qa_info() # triplets是一个list,包含整个bag
            num = len(s.triplets)   # how many sentences support this triplet in the original human-annotated data
            head_tail_relation_list.append([head, tail, rel, question])
        return head_tail_relation_list
        

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            sents = [Sentence(**json.loads(line)) for line in f]
        return cls(sents=sents)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.sents:
                f.write(s.json() + "\n")

    @classmethod
    def load_fewrel(cls, path: str, path_properties: str = "data/wiki_properties.csv"):
        relation_map = load_wiki_relation_map(path_properties)
        groups = {}

        with open(path) as f:
            for i, lst in tqdm(json.load(f).items()):
                for raw in lst:
                    head, tail = mark_fewrel_entity(raw)
                    t = RelationSentence(
                        tokens=raw["tokens"],
                        head=head,
                        tail=tail,
                        label=relation_map[i].pLabel,
                        label_id=i,
                    )
                    groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        return cls(sents=sents)

    @classmethod
    def load_wiki(cls, path: str, path_properties: str = "data/wiki_properties.csv"):
        relation_map = load_wiki_relation_map(path_properties)
        sents = []
        with open(path) as f:
            ds = WikiDataset(
                mode="train", data=json.load(f), pid2vec=None, property2idx=None
            )
            for i in tqdm(range(len(ds))):
                triplets = ds.load_edges(i)
                triplets = [t for t in triplets if t.label_id in relation_map.keys()]
                for t in triplets:
                    t.label = relation_map[t.label_id].pLabel
                if triplets:
                    # ZSBERT only includes first triplet in each sentence
                    for t in triplets:
                        t.zerorc_included = False
                    triplets[0].zerorc_included = True

                    s = Sentence(triplets=triplets)
                    sents.append(s)

        data = cls(sents=sents)
        counter = Counter(t.label for s in data.sents for t in s.triplets)
        threshold = sorted(counter.values())[-113]  # Based on ZSBERT data stats
        labels = [k for k, v in counter.items() if v >= threshold]
        data = data.filter_labels(labels)
        return data

    def filter_labels(self, labels: List[str]):
        label_set = set(labels)
        sents = []
        for s in self.sents:
            triplets = [t for t in s.triplets if t.label in label_set]
            if triplets:
                s = s.copy(deep=True)
                s.triplets = triplets
                sents.append(s)
        return Dataset(sents=sents)

    def train_test_split(self, test_size: int, random_seed: int, by_label: bool):
        random.seed(random_seed)

        if by_label:
            labels = self.get_labels()
            labels_test = random.sample(labels, k=test_size)
            labels_train = sorted(set(labels) - set(labels_test))
            sents_train = self.filter_labels(labels_train).sents
            sents_test = self.filter_labels(labels_test).sents
        else:
            sents_train = [s for s in self.sents]
            sents_test = random.sample(self.sents, k=test_size)

        banned = set(s.text for s in sents_test)  # Prevent sentence overlap
        sents_train = [s for s in sents_train if s.text not in banned]
        assert len(self.sents) == len(sents_train) + len(sents_test)
        return Dataset(sents=sents_train), Dataset(sents=sents_test)

    def analyze(self):
        info = dict(
            sents=len(self.sents),
            unique_texts=len(set(s.triplets[0].text for s in self.sents)),
            lengths=str(Counter(len(s.triplets) for s in self.sents)),
            labels=len(self.get_labels()),
        )
        print(json.dumps(info, indent=2))
