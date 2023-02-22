from pathlib import Path
from typing import Dict, List, Tuple

from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer

from transformer_base import run_summarization
from utils import RelationData, RelationSentence


class Encoder(BaseModel):
    def encode_x(self, x: str) -> str:
        raise NotImplementedError

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        raise NotImplementedError

    def decode(self, x: str, y: str) -> RelationSentence:
        raise NotImplementedError

    def decode_x(self, x: str) -> str:
        raise NotImplementedError

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

    def encode_to_line(self, sent: RelationSentence) -> str:
        raise NotImplementedError

    def decode_from_line(self, line: str) -> RelationSentence:
        raise NotImplementedError

    def parse_line(self, line: str) -> Tuple[str, str]:
        raise NotImplementedError


class GenerateEncoder(Encoder):
    def encode_x(self, r: str) -> str:
        return f"Relation : {r} ."

    def decode_x(self, text: str) -> str:
        return text.split("Relation : ")[-1][:-2]

    def encode_triplet(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Context : {sent.text} Head Entity : {s} , Tail Entity : {o} ."

    def decode_triplet(self, text: str, label: str) -> RelationSentence:
        front, back = text.split(" Head Entity : ")
        _, context = front.split("Context : ")
        head, back = back.split(" , Tail Entity : ")
        tail = back[:-2]
        return RelationSentence.from_spans(context, head, tail, label)

    def encode_y(self, sent: RelationSentence) -> str:
        return self.encode_x(sent.label) + " " + self.encode_triplet(sent)

    def decode_y(self, text: str, label: str) -> RelationSentence:
        del label
        front, back = text.split(" . Context : ")
        label = self.decode_x(front + " .")
        return self.decode_triplet("Context : " + back, label)

    def decode(self, x: str, y: str) -> RelationSentence:
        r = self.decode_x(x)
        sent = self.decode_y(y, r)
        return sent

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.label)
        y = self.encode_y(sent)
        return x, y

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return y + "\n"

    def parse_line(self, line: str) -> Tuple[str, str]:
        return "", line.strip()


class HeadTailGenerateEncoder(Encoder):
    """
    Head Entity : <head> , Tail Entity : <tail> . Context : <sentence> .
    x = Head Entity : <head> , Tail Entity : <tail> .
    y = Head Entity : <head> , Tail Entity : <tail> . Context : <sentence> .
    In fact, I donot need to extract head and tail, it is the input...
    Follow the RelationPrompt, save all properties into RelationSentence

    in this solution, we do not need to parse head, tail, and label.
    Only sentence in Context
    """
    def decode_x(self, text: str) -> str:
        front, back = text.split(" , Tail Entity : ")
        tail = back[:-2]
        head = front.split("Head Entity : ")[-1]

        return head, tail

    def decode_triplet(self, context: str, head: str, tail: str, label: str) -> RelationSentence:
        # text is already the sentence
        return RelationSentence.from_spans(context, head, tail, label)

    def decode_y(self, text: str, head: str, tail: str, label: str) -> RelationSentence:
        front, back = text.split(" . Context : ")
        # head, tail = self.decode_x(front + " .")
        context = back[:-2]
        return self.decode_triplet(context, head, tail, label)

    def decode(self, y: str, head: str, tail: str, label: str) -> RelationSentence:
        # The label from DS is just added
        sent = self.decode_y(y, head, tail, label)
        return sent

    def decode_from_line(self, line: str) -> RelationSentence:
        # Existing error, no use this now.
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def encode_y(self, x: str, sen: str) -> str:
        return f"{x} Context : {sen} ." 

    def encode_x(self, head: str, tail: str) -> str:
        return f"Head Entity : {head} , Tail Entity : {tail} ."

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        # The head and tail here is List[int]
        # use set_tuple
        s, r, o = sent.as_tuple()
        x = self.encode_x(s, o)
        y = self.encode_y(x, sent.text)
        return x, y

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return y + "\n"

    def parse_line(self, line: str) -> Tuple[str, str]:
        return "", line.strip()


class HeadTailGenerateWithPrefixEncoder(Encoder):
    """
    Task: Write a sentence that containing following two entities.
    Head Entity : <head> , Tail Entity : <tail> . Context : <sentence> .
    x = Head Entity : <head> , Tail Entity : <tail> .
    y = Head Entity : <head> , Tail Entity : <tail> . Context : <sentence> .
    In fact, we do not need to extract head and tail, it is the input...
    Follow the RelationPrompt, save all properties into RelationSentence

    in this solution, we do not need to parse head, tail, and label.
    Only sentence in Context

    2022-10-10
    We can change the prefix into some samples
    """
    def decode_x(self, text: str) -> str:
        front, back = text.split(" , Tail Entity : ")
        tail = back[:-2]
        head = front.split("Head Entity : ")[-1]

        return head, tail

    def decode_triplet(self, context: str, head: str, tail: str, label: str) -> RelationSentence:
        # text is already the sentence
        return RelationSentence.from_spans(context, head, tail, label)

    def decode_y(self, text: str, head: str, tail: str, label: str) -> RelationSentence:
        front, back = text.split(" . Context : ")
        context = back[:-2]
        return self.decode_triplet(context, head, tail, label)

    def decode(self, y: str, head: str, tail: str, label: str) -> RelationSentence:
        # The label from DS is just added
        print("the generated context: ", y, "head", head, "tail", tail)
        sent = self.decode_y(y, head, tail, label)
        return sent

    def decode_from_line(self, line: str) -> RelationSentence:
        # Existing error, no use this now.
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def encode_y(self, x: str, sen: str) -> str:
        return f"{x} Context : {sen} ."

    def encode_x(self, s: str, o: str) -> str:
        prefix = "Write a sentence that containing following two entities. "
        return f"{prefix} Head Entity : {s} , Tail Entity : {o} ."

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        # The head and tail here is List[int]
        # use set_tuple
        s, r, o = sent.as_tuple()
        x = self.encode_x(s, o)
        y = self.encode_y(x, sent.text)
        return x, y

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return y + "\n"

    def parse_line(self, line: str) -> Tuple[str, str]:
        return "", line.strip()


class TripletGenerateEncoder(Encoder):
    """
    Relation : <rel> , Head Entity : <head> , Tail Entity : <tail> . Context : <sentence> .
    x = Relation : <rel> , Head Entity : <head> , Tail Entity : <tail> .
    y = Relation : <rel> , Head Entity : <head> , Tail Entity : <tail> . Context : <sentence> .
    In fact, I donot need to extract rel, head and tail, it is the input...
    Follow the RelationPrompt, save all properties into RelationSentence

    in this solution, we do not need to parse head, tail, and label.
    Only sentence in Context
    """
    def decode_x(self, text: str) -> str:
        front, back = text.split(" , Tail Entity : ")
        tail = back[:-2]
        head = front.split("Head Entity : ")[-1]

        return head, tail

    def decode_triplet(self, context: str, head: str, tail: str, label: str) -> RelationSentence:
        # text is already the sentence
        return RelationSentence.from_spans(context, head, tail, label)

    def decode_y(self, text: str, head: str, tail: str, label: str) -> RelationSentence:
        front, back = text.split(" . Context : ")
        context = back[:-2]
        return self.decode_triplet(context, head, tail, label)

    def decode(self, y: str, head: str, tail: str, label: str) -> RelationSentence:
        # The label from DS is just added
        sent = self.decode_y(y, head, tail, label)
        return sent

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        # We should provode line and triplets
        return self.decode(x, y)

    def encode_y(self, x: str, sen: str) -> str:
        return f"{x} Context : {sen} ." 

    def encode_x(self, s: str, o: str, r: str, dataset_name=None) -> str:
        # We should normalize the relation
        # For nyt10m, we just split "/" and replace "_" with space
        # r = r.replace("_", " ").replace("/", " / ")
        # v2: just use the last relation name, replace _ with space
        #if dataset_name is None:
        #    # nyt10m default
        #    r = r.split("/")[-1].replace("_", " ")
        #if dataset_name == "re-tacred":
        # These operation are for re-tacred
        r = r.replace("per:", "person - ")
        r = r.replace("org:", "organization - ")
        r = r.replace("/", " or ")
        r = r.replace("_", " ")
        r = r.replace("-", " - ") # This is for semeval
        # 其实就在这边添加一个Relation Words，根据relation，从dict中选择，生成即可。
        return f"Relation : {r} , Head Entity : {s} , Tail Entity : {o} ."


    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        # The head and tail here is List[int]
        # use set_tuple
        s, r, o = sent.as_tuple()
        x = self.encode_x(s, o, r)
        y = self.encode_y(x, sent.text)
        return x, y

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return y + "\n"

    def parse_line(self, line: str) -> Tuple[str, str]:
        return "", line.strip()


class TemplateGenerateEncoder(Encoder):
    """
    Template is the description from head, tail and relation.
    We need to map the triplet into a template. 
    Unfinished.
    """
    def decode_x(self, text: str) -> str:
        front, back = text.split(" , Tail Entity : ")
        tail = back[:-2]
        head = front.split("Head Entity : ")[-1]

        return head, tail

    def decode_triplet(self, context: str, head: str, tail: str, label: str) -> RelationSentence:
        # text is already the sentence
        return RelationSentence.from_spans(context, head, tail, label)

    def decode_y(self, text: str, head: str, tail: str, label: str) -> RelationSentence:
        front, back = text.split(" . Context : ")
        context = back[:-2]
        return self.decode_triplet(context, head, tail, label)

    def decode(self, y: str, head: str, tail: str, label: str) -> RelationSentence:
        # The label from DS is just added
        sent = self.decode_y(y, head, tail, label)
        return sent

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        # We should provode line and triplets
        return self.decode(x, y)

    def encode_y(self, x: str, sen: str) -> str:
        return f"{x} Context : {sen} ." 

    def encode_x(self, s: str, o: str, r: str, dataset_name=None) -> str:
        
        # 其实就在这边添加一个Relation Words，根据relation，从dict中选择，生成即可。
        return f"Relation : {r} , Head Entity : {s} , Tail Entity : {o} ."


    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        # The head and tail here is List[int]
        # use set_tuple
        s, r, o = sent.as_tuple()
        x = self.encode_x(s, o, r)
        y = self.encode_y(x, sent.text)
        return x, y

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return y + "\n"

    def parse_line(self, line: str) -> Tuple[str, str]:
        return "", line.strip()


class ExtractEncoder(Encoder):
    def encode_x(self, text: str) -> str:
        return f"Context : {text}"

    def decode_x(self, x: str) -> str:
        return x.split("Context : ")[-1]

    def encode_y(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Head Entity : {s} , Tail Entity : {o} , Relation : {r} ."

    def decode_y(self, x: str, y: str) -> RelationSentence:
        context = self.decode_x(x)
        front, label = y.split(" , Relation : ")
        label = label[:-2]
        front, tail = front.split(" , Tail Entity : ")
        _, head = front.split("Head Entity : ")
        return RelationSentence.from_spans(context, head, tail, label)

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"Head Entity : {head} , Tail Entity : {tail} , Relation :"

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_y(sent)
        return x, y

    def decode(self, x: str, y: str) -> RelationSentence:
        return self.decode_y(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return run_summarization.encode_to_line(x, y)

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def parse_line(self, line: str) -> Tuple[str, str]:
        return run_summarization.decode_from_line(line)


def test_encoders(
    paths: List[str] = [
        "outputs/data/zsl/wiki/unseen_5_seed_0/train.jsonl",
        "outputs/data/zsl/fewrel/unseen_5_seed_0/train.jsonl",
    ],
    print_limit: int = 4,
    encoder_names: List[str] = ["generate", "extract"],
    limit: int = 1000,
):
    encoders = {k: select_encoder(k) for k in encoder_names}

    for p in paths:
        data = RelationData.load(Path(p))
        _, data = data.train_test_split(min(limit, len(data.sents)), random_seed=0)

        for name, e in tqdm(list(encoders.items())):
            num_fail = 0
            print(dict(name=name, p=p))
            for s in data.sents:
                encoded = e.encode_to_line(s)
                x, y = e.parse_line(encoded)
                decoded: RelationSentence = e.safe_decode(x, y)

                if decoded.as_tuple() != s.as_tuple():
                    if num_fail < print_limit:
                        print(dict(gold=s.as_tuple(), text=s.text))
                        print(dict(pred=decoded.as_tuple(), text=decoded.text))
                        print(dict(x=x, y=y, e=decoded.error))
                        print()
                    num_fail += 1

            print(dict(success_rate=1 - (num_fail / len(data.sents))))
            print("#" * 80)


def select_encoder(name: str) -> Encoder:
    mapping: Dict[str, Encoder] = dict(
        extract=ExtractEncoder(),
        generate=GenerateEncoder(),
        head_tail_generate=HeadTailGenerateEncoder(),     # new added
        head_tail_generate_with_prefix=HeadTailGenerateWithPrefixEncoder(),     # new added
        triplet_generate=TripletGenerateEncoder(),
        triplet=TripletGenerateEncoder(),
        template=TemplateGenerateEncoder()
    )
    encoder = mapping[name]
    return encoder


def test_entity_prompts(
    path: str = "outputs/data/zsl/wiki/unseen_10_seed_0/test.jsonl", limit: int = 100
):
    def tokenize(text: str, tok) -> List[str]:
        return tok.convert_ids_to_tokens(tok(text, add_special_tokens=False).input_ids)

    data = RelationData.load(Path(path))
    e = ExtractEncoder()
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    print(tokenizer)
    for i, s in enumerate(tqdm(data.sents[:limit])):
        head, label, tail = s.as_tuple()
        x, y = e.encode(s)
        prompt = e.encode_entity_prompt(head, tail)
        tokens_y = tokenize(y, tokenizer)
        tokens_prompt = tokenize(prompt, tokenizer)
        assert tokens_y[: len(tokens_prompt)] == tokens_prompt
        if i < 3:
            print(tokens_y)


if __name__ == "__main__":
    Fire()
