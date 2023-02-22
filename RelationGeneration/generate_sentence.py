import os
import random
import argparse
from pathlib import Path
from wrapper import Generator, Extractor, Dataset, TripletDataset

"""
2022-09-21, Junjie Yu, Soochow University
Train the PLM by DS corpus, and then generate synthetic instances (head, tail, relation and sentences)
Based "head + tail to sentence", we fine-tune the GPT-2 by DS data.
Goal: train a generator which can generate synthetic sentences by taking as input two entities.
A generation solution rather than matching solution used in conventional distantly supervision relation extraction.

2022-10-06, Junjie Yu, Soochow University
Wrap all the generator should do:
1. training the model by NonNA triplets with multiple supporting sentences.
2. Do quality test on the synthetic sentences:
    2.1. train the model by synthetic data and output results
"""


class GenerateData():
    """
    Using fine-tuned GPT-2 to generate synthetic sentences by specific prompt
    """
    def __init__(self, args, dir_kwargs, file_kwargs):
        """
        dir_kwargs: dictionary of keyword arguments about input and output data directory
        file_kwargs: dictionary
        data_limit is the number of training examples to train for debugging purposes
            default is -1, means use all sentences
        ONLY load processed parameters, like train file
        """
        self.args = args
        # process the datasets
        self.input_file = file_kwargs["input_file"]
        self.output_file = file_kwargs["output_file"]

        # the model dir after fine-tuning the GPT
        self.generator_dir = dir_kwargs['generator_dir']

    def truncate_data(self, path_in:str, limit:int, path_out:str):
        if os.path.exists(path_out):
            print(f"Truncated data exists: {path_out}")
            return
        # Use a subset of data for quick demo on Colab
        data = Dataset.load(path_in)
        random.seed(0)  # here, the seed is not very important, it is only for debugging
        random.shuffle(data.sents)
        data.sents = data.sents[:limit]
        data.save(path_out)
    
    def generate(self, model_kwargs, training_kwargs, debug_num=-1):
        """debug_num: the debug of how many triplets"""
        # model_kwargs = dict(batch_size=16, grad_accumulation=4)  # For 1080ti
        # save_dir = os.path.join(self.save_dir, training_kwargs['encoder_name'])
        encoder_name = training_kwargs['encoder_name']
        # 默认是选取训练过的模型，路径不一样。目前取消了这个
        if False: # self.args.use_original_model:
            # the save_dir is actually the model
            print(dir_kwargs['model_dir'])
            generator = Generator(
                load_dir=dir_kwargs['model_dir'],
                save_dir=dir_kwargs['model_dir'],   # in select_model, add a subdir "model"
                model_kwargs=model_kwargs,
                encoder_name=encoder_name
            )
        else:
            # For generate data, the load_dir is the model in generator dir 
            generator = Generator(
                load_dir=str(Path(self.generator_dir) / "model"),
                save_dir=self.generator_dir,    # This will not used in generation.
                model_kwargs=model_kwargs,
                encoder_name=encoder_name
            )
        # generate synthetic sentences by head, tail, relation under specific prompt
        print(f"Use triplets from {self.input_file} to generate sentences.")
        # only load triplet and number
        # do not need Dataset, or create a Special Dataset for untrain data.
        # num=5 is a default number, actually it depends on the supporting sentences in original dataset
        input_data = TripletDataset.load(self.input_file, num=5)
        # heads_tails_relations_nums = input_data.get_heads_tails_relations()
        if encoder_name == "triplet":
            input_triplets = input_data.get_head_tail_rel_num()
            if debug_num > 0:
                input_triplets = input_triplets[:debug_num]
            generator.generate_by_triplet(input_triplets, path_out=self.output_file)
        elif generator.generate_by_template():
            pass
        else:
            print(f"Error encoder name: {encoder_name}")
        print("End generation.")


class GenerationArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        return self.parser.parse_args()

    def add_arguments(self):
        self.parser.add_argument(
            "--prompt_name",
            default=None,
            type=str,
            required=True,
            help="[head_tail_generate, triplet_generate]",
        )
        self.parser.add_argument(
            "--pretrained_dir",
            default=None,
            type=str,
            required=True,
            help="the path to pre-trained model",
        )
        self.parser.add_argument("--use_original_model",
                            action="store_true",
                            help="Whether to use the original model, like GPT-2")
        self.parser.add_argument(
            "--save_dir",
            default=None,
            type=str,
            required=True,
            help="For training, the path to save model or data. For generation, it is the path for loading the pretrained model and save synthetic data.",
        )
        self.parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="path to dir containing train, val, test file",
        )
        self.parser.add_argument(
            "--train_file",
            default=None,
            type=str,
            required=True,
            help="train file path and name.",
        )
        self.parser.add_argument(
            "--val_file",
            default=None,
            type=str,
            required=True,
            help="val file path and name.",
        )
        self.parser.add_argument(
            "--test_file",
            default=None,
            type=str,
            required=True,
            help="test file path and name.",
        )
        self.parser.add_argument(
            "--data_limit",
            default=-1,
            type=int,
            required=False,
            help="Use the number of training examples to debug",
        )
        self.parser.add_argument(
            "--seed",
            default=0,
            type=int,
            required=True,
            help="init seed",
        )


if __name__ == "__main__":
    #parser = GenerationArguments()
    #parser.add_arguments()
    #args = parser.parse_args()

    parser = argparse.ArgumentParser(description="arguments to generate sentences for a dataset.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--scale", type=str, default="1.0", required=True, help="1.0 means use all training data. Others are 0.1, 0.2, 0.5")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt", type=str, default="template", choices=["triplet", "template", "qa"])
    parser.add_argument("--debug_num", type=int, default=-1)
    args = parser.parse_args()

    # generator_dir = f"/data/jjyu/RE_Sentence_Generation/Fine_tuned_model/{args.dataset}/{args.gpt_size}/{args.prompt}/generator"
    gpt_size = "gpt2-large"
    generator_dir = f"/data/jjyu/RE_Sentence_Generation/Fine_tuned_model/{args.dataset}/scale{args.scale}_seed{args.seed}/{gpt_size}/{args.prompt}/generator"

    # For generation, seed is not important as we do it once
    training_kwargs = dict(seed=42, encoder_name=args.prompt)
    model_kwargs = dict(batch_size=32, grad_accumulation=1)

    data_dir = f"../DatasetForGeneration/{args.dataset}"
    # We need input triplet and its number of supporting sentences in original dataset
    input_file = f"{data_dir}/untrain_triplet.txt"  # triplet:[inst, inst, ...]
    output_file = f"{generator_dir}/output_data/synthetic_sentence_for_untrain_triplet.json"

    if args.debug_num > 0:
        output_file = f"{generator_dir}/output_data/synthetic_sentence_for_untrain_triplet_{args.debug_num}.json"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    file_kwargs = dict(input_file=input_file,
                       output_file=output_file)

    dir_kwargs = dict(
        generator_dir=generator_dir,
    )
    print(training_kwargs)
    print(model_kwargs)
    print(file_kwargs)
    print(dir_kwargs)
    print("----------------------------------------------------------------")
    data_generator = GenerateData(args, dir_kwargs, file_kwargs)
    data_generator.generate(model_kwargs, training_kwargs, debug_num=args.debug_num)

