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

from dataset import Sentence, Dataset

def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b

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
    """
    For training, fit() will invoke the CLM training.
    For generation, it directly generates.
    """
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
        lines = [encoder.encode_to_line(t) for s in data.sents for t in s.triplets]     # 得到prompt+回答
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
        data_dev = Dataset.load(path_dev)
        # write data，包括encoding，即把语料转换成指定的prompt的形式
        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)     # CLM的训练只要输入训练集和测试集就行，因此，跟下面的generate们没关系
        delete_checkpoints(model.model_dir)

    # 这些函数，会去modeling里调用相似名字的函数。但是，这些函数又是什么时候用的？？？上面的fit直接就跑了呀。
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
        add number item in heads_tails_relations
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
            triplets, raw = model.generate_by_triplet(head, tail, relation, num, pipe=pipe)
            for t in triplets:
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
