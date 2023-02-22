import json
import random
from collections import Counter
from pathlib import Path
from typing import List

import torch
from fire import Fire
from pydantic.main import BaseModel
from tqdm import tqdm

from generation import LabelConstraint, TripletSearchDecoder
from modeling import (NewRelationExtractor, RelationGenerator, RelationModel,
                      select_model)
from utils import (RelationSentence, WikiDataset, delete_checkpoints,
                   load_wiki_relation_map, mark_fewrel_entity)
import sys
print(sys.path)
sys.path.insert(0, "/home/jjyu/GenerationForRE" )
from REUtils.data_io import load_data_from_file


def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b


def convert_RE_data(path_data, rel2template):
    """
    The format of re-tacred and semeval is unmatched with the input of RelationPrompt.
    Add template
    """
    data = []
    with open(path_data) as reader:
        for line in reader.readlines():
            data.append(json.loads(line.strip()))
    triplets = {}
    new_data = []
    for inst in data:
        # print(inst)
        tokens = inst['token']
        head_start, head_end = inst['h']['pos']
        tail_start, tail_end = inst['t']['pos']
        head_name = inst['h']['name']
        tail_name = inst['t']['name']
        # the re-tacred also use token-based position
        head_index = [x for x in range(head_start, head_end)]
        tail_index = [x for x in range(tail_start, tail_end)]

        label = inst['relation']
        assert label in rel2template
        template = rel2template[label]
        inst = {
            "tokens": tokens,
            "head": head_index,
            "tail": tail_index,
            "head_name": head_name,
            "tail_name": tail_name,
            "label": label,
            "template": template
        }
        new_data.append(inst)   # 这个先不用了，直接用1-N的triplets
        triplet = f"{head_name}_{tail_name}_{label}"
        if triplet not in triplets:
            triplets[triplet] = [inst]
        else:
            triplets[triplet].append(inst)
    return triplets


class HeadTailRelNum(BaseModel):
    """不需要"""
    head_name: str
    tail_name: str
    relation: str
    number: int

    def as_triplet_prompt(self,):
        """感觉不需要，这个是encoder做的事"""
        pass


class TripletDataset(BaseModel):
    """
    The data structure for generation: triplet -> sentence
    """
    head_tail_rel_num: List

    @classmethod
    def convert(cls, line, num):
        """convert untrain triplet input into HeadTailRelNum"""
        inst = json.loads(line)
        triplet = inst["triplet"]
        head, tail, rel = triplet.split("#-#")
        if "insts" in inst:
            num = len(inst['insts'])
        # return {"head": head, "tail": tail, "rel": rel, "num": num}
        return [head, tail, rel, num]

    @classmethod
    def load(cls, path: str, num: int):
        """num is the default number of required sentences for each triplet to generate"""
        with open(path) as f:
            head_tail_rel_num = [cls.convert(line, num) for line in f]
        return cls(head_tail_rel_num=head_tail_rel_num)
    

    def get_head_tail_rel_num(self) -> List:
        # Maybe we can build the prompt, the x
        return self.head_tail_rel_num

    def get_head_tail_rel_template_num(self, rel2template) -> List:
        """construct the template for each triplet, with a mapping dict
        """
        head_tail_rel_template_num = []
        for head, tail, rel, num in self.head_tail_rel_num:
            assert rel in rel2template
            template = rel2template[rel]
            head_tail_rel_template_num.append([head, tail, rel, template, num])
        return head_tail_rel_template_num



class GeneratedDataset(BaseModel):
    """
    The data structure for save generated sentences
    {triplet: head_tail_rel, num: num, insts: [inst1, inst2, ...,]}
    """
    # 暂时也不用了，就生成后在convert中处理吧
    pass


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
    
    @classmethod
    def load_with_convert(cls, path_data: str, rel2template):
        """We need to convert our dataset into target format.
        This is for re-tacred.
        We can refer the follow function load_fewrel, load_wiki."""
        data = convert_RE_data(path_data, rel2template)    # load data and add column template
        
        # 我们将triplet改成head, tail, relation, number
        sents = [Sentence(**{"triplets": list(insts)}) for insts in data.values()]
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


def write_data_splits(
    path_in: str,
    mode: str,
    folder_out: str = "outputs/data/splits/zero_rte",
    num_dev_labels: int = 5,
    num_test_labels: List[int] = [5, 10, 15],
    seeds: List[int] = [0, 1, 2, 3, 4],
):
    for n in num_test_labels:
        for s in seeds:
            if mode == "fewrel":
                data = Dataset.load_fewrel(path_in)
            elif mode == "wiki":
                data = Dataset.load_wiki(path_in)
            else:
                raise ValueError()

            train, test = data.train_test_split(
                test_size=n, random_seed=s, by_label=True
            )
            train, dev = train.train_test_split(
                test_size=num_dev_labels, random_seed=s, by_label=True
            )
            del data

            for key, data in dict(train=train, dev=dev, test=test).items():
                name = f"unseen_{n}_seed_{s}"
                path = Path(folder_out) / Path(path_in).stem / name / f"{key}.jsonl"
                data.save(str(path))
                print(dict(key=key, labels=len(data.get_labels()), path=path))


class Generator(BaseModel):
    # These class variables will be initialized as the __init__ function in BaseModel
    load_dir: str
    save_dir: str
    num_gen_per_label: int = 5  # default 250, too large for DS bag
    model_name: str = "generate"
    encoder_name: str = "generate"  # This will be replaced by our encoder_name
    model_kwargs: dict = {}

    def get_model(self) -> RelationModel:
        model = select_model(
            name=self.model_name,   # this arg for select model, follows for set model
            encoder_name=self.encoder_name,     # the encoder name is important
            model_dir=str(Path(self.save_dir) / "model"),
            model_name=self.load_dir,
            data_dir=str(Path(self.save_dir) / "data"),
            do_pretrain=False,
            **self.model_kwargs,
        )
        return model

    def write_data(self, data: Dataset, name: str) -> str:
        model = self.get_model()
        path_out = Path(model.data_dir) / f"{name}.txt"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        encoder = model.get_encoder()   # 签名已经确定了这边是什么encoder，比如generate
        # 这里最终只保留sents? data.sents又是啥，猜测是sentence
        # encode_to_line会把t转换成指定的prompt形式
        # 在原论文中，这个就是Relation: <rel> . Context : {sent.text} Head Entity : {s} , Tail Entity : {o} .
        # 那我的修改就在这里，即把encode_to_entity_pair_line
        # 即 Head Entity : {s} , Tail Entity : {o} . Context : {sent.text} .
        # Template : template . Context : X
        lines = [encoder.encode_to_line(t) for s in data.sents for t in s.triplets]     # 得到prompt+回答
        random.seed(model.random_seed)
        random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))
        return str(path_out)

    def fit(self, path_train: str, path_dev: str, path_template: str):
        """
        train on path_train with path_template to build input sequence.
        """
        model = self.get_model()
        if Path(model.model_dir).exists():
            return
        # {"relation": X, "template": Y, "QA": Z}
        info_data = load_data_from_file(path_template)
        rel2template = {}
        for inst in info_data:
            relation = inst['relation']
            template = inst['template']
            rel2template[relation] = template
        # data_train = Dataset.load(path_train)
        # we need to convert RE data into relationprompt format
        data_train = Dataset.load_with_convert(path_train, rel2template)
        data_dev = Dataset.load_with_convert(path_dev, rel2template)
        # write data，包括encoding，即把语料转换成指定的prompt的形式
        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)     # CLM的训练只要输入训练集和测试集就行，因此，跟下面的generate们没关系
        delete_checkpoints(model.model_dir)

    def generate(self, labels: List[str], path_out: str):
        if Path(path_out).exists():
            return

        model = self.get_model()
        pipe = model.make_pipe()
        groups = {}
        assert isinstance(model, RelationGenerator)
        for relation in tqdm(labels):
            triplets, raw = model.generate(relation, self.num_gen_per_label, pipe=pipe)
            for t in triplets:
                groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        data = Dataset(sents=sents)
        data.save(path_out)

    def generate_by_head_tail(self, heads_tails_relations: List, path_out: str):
        """
        Head Tail to Sentence
        """
        if Path(path_out).exists():
            return

        model = self.get_model()
        # what make pipe means
        pipe = model.make_pipe()    # load data from self.model_dir     these args are confused
        groups = {}
        assert isinstance(model, RelationGenerator)
        for head, tail, relation in tqdm(heads_tails_relations):

            triplets, raw = model.generate_by_head_tail(head, tail, relation, self.num_gen_per_label, pipe=pipe)
            for t in triplets:
                groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        data = Dataset(sents=sents)
        data.save(path_out)

    def generate_by_head_tail_with_prefix(self, heads_tails_relations: List, path_out: str):
        """
        Head Tail to Sentence
        """
        if Path(path_out).exists():
            return

        model = self.get_model()
        # what make pipe means
        pipe = model.make_pipe()    # load data from self.model_dir
        groups = {}
        assert isinstance(model, RelationGenerator)
        for head, tail, relation in tqdm(heads_tails_relations):

            triplets, raw = model.generate_by_head_tail_with_prefix(head, tail, relation, self.num_gen_per_label, pipe=pipe)
            for t in triplets:
                groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        data = Dataset(sents=sents)
        data.save(path_out)

    def generate_by_triplet(self, heads_tails_relations_nums: List, path_out: str):
        """
        Relation Head Tail to Sentence
        Add number of sentences for each triplet to generate.
        """
        if Path(path_out).exists():
            print(f"{path_out} exist!!!")
            return

        model = self.get_model()
        pipe = model.make_pipe()
        groups = {}
        assert isinstance(model, RelationGenerator)
        for head, tail, relation, num in tqdm(heads_tails_relations_nums):
            # set the num_gen_per_label here
            # 我感觉在这边就准备好prompt，而model里就只有一个generate就行。因为generate都是一样的。
            # 但是,encoder是放在model里了，确实放model里比较合理。
            triplets, raw = model.generate_by_triplet(head, tail, relation, num, pipe=pipe)
            # 总觉得下面这步是多余的，我这边的输出triplets，直接就是sents
            # 稍微有点区别，generate输出的是[RelationSentence]
            # 下面的是[Sentence]，而Sentence是List[RelationSentence]
            # 这边的话，则是每个Sentence下只有一个元素的list
            # 最后的Dataset则是List[Sentence]
            for t in triplets:
                # get the value of t.text, if not exist, set t.text:[], and return the value.
                # t.text is the " ".join(tokens), a @property in Sentence
                # text as key, all same sentences as value
                # 我们不需要用text作为key等，直接上一步得到的数据做一些处理就好。但，影响不大，我后面做处理的。
                groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        data = Dataset(sents=sents)
        data.save(path_out)

    def generate_by_template(self, heads_tails_rels_templates_nums: List, path_out: str):
        """
        Template to Sentence
        Add number of sentences for each triplet to generate.
        """
        if Path(path_out).exists():
            print(f"{path_out} exist!!!")
            return

        model = self.get_model()
        pipe = model.make_pipe()
        groups = {}
        assert isinstance(model, RelationGenerator)
        for head, tail, rel, template, num in tqdm(heads_tails_rels_templates_nums):
            # set the num_gen_per_label here
            # 我感觉在这边就准备好prompt，而model里就只有一个generate就行。因为generate都是一样的。
            # 但是,encoder是放在model里了，确实放model里比较合理。
            triplets, raw = model.generate_by_template(head, tail, rel, template, num, pipe=pipe)
            # 总觉得下面这步是多余的，我这边的输出triplets，直接就是sents
            # 稍微有点区别，generate输出的是[RelationSentence]
            # 下面的是[Sentence]，而Sentence是List[RelationSentence]
            # 这边的话，则是每个Sentence下只有一个元素的list
            # 最后的Dataset则是List[Sentence]
            for t in triplets:
                # get the value of t.text, if not exist, set t.text:[], and return the value.
                # t.text is the " ".join(tokens), a @property in Sentence
                # text as key, all same sentences as value
                # 我们不需要用text作为key等，直接上一步得到的数据做一些处理就好。但，影响不大，我后面做处理的。
                groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        data = Dataset(sents=sents)
        data.save(path_out)

    def generate_by_qa(self, heads_tails_rels_questions: List, path_out: str):
        """
        Relation Head Tail to Sentence
        Add number of sentences for each triplet to generate.
        add number item in heads_tails_relations
        """
        if Path(path_out).exists():
            print(f"{path_out} exist!!!")
            return

        model = self.get_model()
        pipe = model.make_pipe()
        triplets = []
        assert isinstance(model, RelationGenerator)
        for head, tail, rel, question in tqdm(heads_tails_rels_questions):
            # set the num_gen_per_label here
            # Give 10 candidate answers
            print(head, tail, question)
            candidate_answers = model.generate_by_qa(head, tail, question, num=10, pipe=pipe)
            # For 10 candidate answers, we should put them together
            # 不知道这边生成的什么样子的，我们仅需要triplet, candidate answers，从而来筛选triplets
                # {head: XX, tail: XX, question_rel: XX, answers=[]}    最好加上原来的rel，这样省的后期还要mapping一下,question_rel是用于识别answer是head还是tail
            triplet = {"head": head, "tail": tail, "rel": rel, "question": question, "answers": candidate_answers}
            triplets.append(triplet)
        print(triplets)
        with open(path_out, 'w') as writer:
            for triplet in triplets:
                writer.write(json.dumps(triplet) + '\n')

    # 这个没准就是template的实现？看看，修改
    def generate_by_head_tail_description(self, inputs: List, path_out: str):
        """
        Head Tail Description to Sentence
        inputs: inst items, like head, tail, ...
        """
        if Path(path_out).exists():
            print(f"{path_out} exist!!!")
            return

        model = self.get_model()
        pipe = model.make_pipe()
        groups = {}
        assert isinstance(model, RelationGenerator)
        for head, tail, description, num, rel in tqdm(inputs):
            # set the num_gen_per_label here
            triplets, raw = model.generate_by_head_tail_description(head, tail, description, num*4, rel, pipe=pipe)
            for t in triplets:
                # the sentence as key
                # 这个就是我常做的一个功能，如果不存在key，则初始化一个。否则就不执行初始化的内容。
                groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        # sents is a list containing sentence2info, [Sentence(triplets=[inst1, inst2]), Sentence...]
        # inst1 and inst2 have the same sentence
        # Maybe we should change the key by adding triplet, like head_tail_rel
        data = Dataset(sents=sents)
        data.save(path_out)



class Extractor(BaseModel):
    load_dir: str
    save_dir: str
    model_name: str = "new_extract"
    encoder_name: str = "extract"
    search_threshold: float = -0.9906
    model_kwargs: dict = {}

    def get_model(self) -> RelationModel:
        model = select_model(
            name=self.model_name,
            encoder_name=self.encoder_name,
            model_dir=str(Path(self.save_dir) / "model"),
            model_name=self.load_dir,
            data_dir=str(Path(self.save_dir) / "data"),
            do_pretrain=False,
            **self.model_kwargs,
        )
        return model

    def write_data(self, data: Dataset, name: str) -> str:
        model = self.get_model()
        path_out = Path(model.data_dir) / f"{name}.json"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        encoder = model.get_encoder()
        lines = [encoder.encode_to_line(t) for s in data.sents for t in s.triplets]
        random.seed(model.random_seed)
        random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))
        return str(path_out)

    def fit(self, path_train: str, path_dev: str):
        model = self.get_model()
        if Path(model.model_dir).exists():
            return

        data_train = Dataset.load(path_train)
        data_train = Dataset.load(path_train)
        data_dev = Dataset.load(path_dev)
        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def predict(self, path_in: str, path_out: str, use_label_constraint: bool = True):
        data = Dataset.load(path_in)
        texts = [s.text for s in data.sents]
        model = self.get_model()
        assert isinstance(model, NewRelationExtractor)
        gen = model.load_generator(torch.device("cuda"))
        encoder = model.get_encoder()
        constraint = LabelConstraint(labels=data.get_labels(), tokenizer=gen.tokenizer)
        sents = []

        for i in tqdm(range(0, len(texts), model.batch_size)):
            batch = texts[i : i + model.batch_size]
            x = [encoder.encode_x(t) for t in batch]
            outputs = model.gen_texts(
                x, gen, num_beams=1, save_scores=use_label_constraint
            )
            assert len(outputs) == len(x)

            for i, raw in enumerate(outputs):
                triplet = encoder.safe_decode(x[i], y=raw)
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[i])
                sents.append(Sentence(triplets=[triplet]))

        Dataset(sents=sents).save(path_out)

    def predict_multi(self, path_in: str, path_out: str):
        stem = Path(path_out).stem
        path_raw = path_out.replace(stem, f"{stem}_raw")
        print(dict(predict_multi=locals()))
        data = Dataset.load(path_in)
        model = self.get_model()
        assert isinstance(model, NewRelationExtractor)
        gen = model.load_generator(torch.device("cuda"))
        constraint = LabelConstraint(labels=data.get_labels(), tokenizer=gen.tokenizer)
        searcher = TripletSearchDecoder(
            gen=gen, encoder=model.get_encoder(), constraint=constraint
        )

        sents = [
            Sentence(tokens=s.tokens, triplets=searcher.run(s.text))
            for s in tqdm(data.sents)
        ]
        Dataset(sents=sents).save(path_raw)
        for s in sents:
            s.triplets = [t for t in s.triplets if t.score > self.search_threshold]
        Dataset(sents=sents).save(path_out)

    @staticmethod
    def score(path_pred: str, path_gold: str) -> dict:
        pred = Dataset.load(path_pred)
        gold = Dataset.load(path_gold)
        assert len(pred.sents) == len(gold.sents)
        num_pred = 0
        num_gold = 0
        num_correct = 0

        for i in range(len(gold.sents)):
            num_pred += len(pred.sents[i].triplets)
            num_gold += len(gold.sents[i].triplets)
            for p in pred.sents[i].triplets:
                for g in gold.sents[i].triplets:
                    if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                        num_correct += 1

        precision = safe_divide(num_correct, num_pred)
        recall = safe_divide(num_correct, num_gold)

        info = dict(
            path_pred=path_pred,
            path_gold=path_gold,
            precision=precision,
            recall=recall,
            score=safe_divide(2 * precision * recall, precision + recall),
        )
        return info


def main(
    path_train: str,
    path_dev: str,
    path_test: str,
    save_dir: str,
):
    print(dict(main=locals()))
    generator = Generator(
        load_dir="gpt2",
        save_dir=str(Path(save_dir) / "generator"),
    )
    extractor = Extractor(
        load_dir="facebook/bart-base",
        save_dir=str(Path(save_dir) / "extractor"),
    )

    generator.fit(path_train, path_dev)
    extractor.fit(path_train, path_dev)
    path_synthetic = str(Path(save_dir) / "synthetic.jsonl")
    labels_dev = Dataset.load(path_dev).get_labels()
    labels_test = Dataset.load(path_test).get_labels()
    generator.generate(labels_dev + labels_test, path_out=path_synthetic)

    extractor_final = Extractor(
        load_dir=str(Path(save_dir) / "extractor" / "model"),
        save_dir=str(Path(save_dir) / "extractor_final"),
    )
    extractor_final.fit(path_synthetic, path_dev)

    path_pred = str(Path(save_dir) / "pred.jsonl")
    extractor_final.predict(path_in=path_test, path_out=path_pred)
    results = extractor_final.score(path_pred, path_test)
    print(json.dumps(results, indent=2))
    with open(Path(save_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main_many(data_dir_pattern: str, save_dir: str, **kwargs):
    mode = Path(save_dir).name
    assert mode in ["fewrel", "wiki"]
    records = []

    for path in tqdm(sorted(Path().glob(data_dir_pattern))):
        path_train = path / "train.jsonl"
        path_dev = path / "dev.jsonl"
        path_test = path / "test.jsonl"
        results = main(
            path_train=str(path_train),
            path_dev=str(path_dev),
            path_test=str(path_test),
            save_dir=str(Path(save_dir) / path.name),
            **kwargs,
        )
        records.append(results)

    avg_p = sum([r["precision"] for r in records]) / len(records)
    avg_r = sum([r["recall"] for r in records]) / len(records)
    avg_f = safe_divide(2 * avg_p * avg_r, avg_p + avg_r)
    info = dict(avg_p=avg_p, avg_r=avg_r, avg_f=avg_f)
    print(json.dumps(info, indent=2))


def run_eval(path_model: str, path_test: str, mode: str, limit: int = 0):
    print(dict(run_eval=locals()))
    data = Dataset.load(path_test)
    model = Extractor(load_dir=str(Path(path_model) / "model"), save_dir=path_model)

    if mode == "single":
        data.sents = [s for s in data.sents if len(s.triplets) == 1]
    elif mode == "multi":
        data.sents = [s for s in data.sents if len(s.triplets) > 1]
    else:
        raise ValueError(f"mode must be single or multi")

    if limit > 0:
        random.seed(0)
        random.shuffle(data.sents)
        data.sents = data.sents[:limit]

    path_in = str(Path(path_model) / f"pred_in_{mode}.jsonl")
    path_out = str(Path(path_model) / f"pred_out_{mode}.jsonl")
    data.save(path_in)

    if mode == "single":
        model.predict(path_in, path_out)
    else:
        model.predict_multi(path_in, path_out)

    results = model.score(path_pred=path_out, path_gold=path_in)
    path_results = str(Path(path_model) / f"results_{mode}.json")
    results.update(mode=mode, limit=limit, path_results=path_results)
    print(json.dumps(results, indent=2))
    with open(path_results, "w") as f:
        json.dump(results, f, indent=2)


def run_eval_many(path_model_pattern: str, data_dir: str, **kwargs):
    for path in tqdm(sorted(Path().glob(path_model_pattern))):
        name = path.parts[-2]
        path_test = Path(data_dir) / name / "test.jsonl"
        assert path_test.exists()
        run_eval(path_model=str(path), path_test=str(path_test), **kwargs)


"""
FewRel Dataset

python wrapper.py main \
--path_train outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/train.jsonl \
--path_dev outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/dev.jsonl \
--path_test outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/test.jsonl \
--save_dir outputs/wrapper/fewrel/unseen_10_seed_0

python wrapper.py run_eval \
--path_model outputs/wrapper/fewrel/unseen_10_seed_0/extractor_final \
--path_test outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/test.jsonl \
--mode single

python wrapper.py run_eval \
--path_model outputs/wrapper/fewrel/unseen_10_seed_0/extractor_final \
--path_test outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/test.jsonl \
--mode multi

Wiki-ZSL Dataset

python wrapper.py main \
--path_train outputs/data/splits/zero_rte/wiki/unseen_10_seed_0/train.jsonl \
--path_dev outputs/data/splits/zero_rte/wiki/unseen_10_seed_0/dev.jsonl \
--path_test outputs/data/splits/zero_rte/wiki/unseen_10_seed_0/test.jsonl \
--save_dir outputs/wrapper/wiki/unseen_10_seed_0

python wrapper.py run_eval \
--path_model outputs/wrapper/wiki/unseen_10_seed_0/extractor_final \
--path_test outputs/data/splits/zero_rte/wiki/unseen_10_seed_0/test.jsonl \
--mode single

python wrapper.py run_eval \
--path_model outputs/wrapper/wiki/unseen_10_seed_0/extractor_final \
--path_test outputs/data/splits/zero_rte/wiki/unseen_10_seed_0/test.jsonl \
--mode multi

"""


if __name__ == "__main__":
    Fire()
