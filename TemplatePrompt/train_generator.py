import os
import argparse
import random
from pathlib import Path
from wrapper import Generator, Extractor, Dataset

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

2022-12-14, Junjie Yu, Soochow University
Base triplet training
Input is a dataset name as a folder name that contains train, val, test data. 

2023-02-06, Junjie Yu, Soochow University
We rebuild the scenario and format of datasets.

2023-02-13, Junjie Yu, Soochow University
Add introducing rel2info file, to convert label to template.
Reconstruct a version of PromptGeneration.

"""


class TrainGenerator():
    """
    Fine-tune the GPT-2 with input data with specific prompt
    """
    def __init__(self, dir_kwargs, file_kwargs, data_limit=-1):
        """
        data_limit is the number of training examples to train for debugging purposes
            default is -1, means use all sentences
        model_args: dict with training parameters
            batch_size, grad_accumulation
        training_args: dict with training parameters
            seed, limit
        ONLY load processed parameters, like train file
        """
        # process the datasets
        train_file = file_kwargs["train_file"]
        val_file = file_kwargs["val_file"]
        test_file = file_kwargs["val_file"]
        rel2template_file = file_kwargs["rel2template_file"]
        data_dir = dir_kwargs["data_dir"]

        self.path_to_train = os.path.join(data_dir, train_file)
        self.path_to_val = os.path.join(data_dir, val_file)
        self.path_to_test = os.path.join(data_dir, test_file)
        self.path_to_rel2template = os.path.join(data_dir, rel2template_file)
        self.save_dir = dir_kwargs["output_dir"]
        if data_limit != -1:
            temp_data_dir = os.path.join(data_dir, f"train_{data_limit}")
            path_out_train = os.path.join(temp_data_dir, train_file)
            path_out_val = os.path.join(temp_data_dir, val_file)
            path_out_test = os.path.join(temp_data_dir, test_file)
            self.truncate_data(self.path_to_train, limit=data_limit, path_out=path_out_train)
            self.truncate_data(self.path_to_val, limit=data_limit, path_out=path_out_val)
            self.truncate_data(self.path_to_test, limit=data_limit, path_out=path_out_test)
            
            self.path_to_train = path_out_train
            self.path_to_val = path_out_val
            self.path_to_test = path_out_test

            self.save_dir= os.path.join(dir_kwargs["output_dir"], f"train_{data_limit}")

    def truncate_data(self, path_in:str, limit:int, path_out:str):
        # Use a subset of data for quick demo on Colab
        data = Dataset.load(path_in)
        random.seed(0)  # here, the seed is not very important, it is only for debugging
        random.shuffle(data.sents)
        data.sents = data.sents[:limit]
        data.save(path_out)
    
    def train(self, model_kwargs, training_kwargs):
        # model_kwargs = dict(batch_size=16, grad_accumulation=4)  # For 1080ti
        save_dir = os.path.join(self.save_dir, training_kwargs['encoder_name'])

        generator = Generator(
            load_dir=dir_kwargs['model_dir'],
            save_dir=str(Path(save_dir) / "generator"),
            model_kwargs=model_kwargs,
            encoder_name=training_kwargs['encoder_name'],   # This encoder name is important which controls the format of prompt
        )

        # Fine-tuning the PLM with corpus
        print(f"Start fine-tuning the PLM on {self.path_to_train}")
        generator.fit(self.path_to_train, self.path_to_val, self.path_to_rel2template)
        print("End fine-tuning")


if __name__ == "__main__":
    limit = -1
    parser = argparse.ArgumentParser(description="arguments to train a generator.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--scale", type=str, default="1.0", required=True, help="1 means use all training data. Others are 0.1, 0.2, 0.5")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt", type=str, default="template")
    args = parser.parse_args()

    training_kwargs = dict(seed=args.seed, encoder_name=args.prompt)
    
    # generator training is on V100-32G
    model_kwargs = dict(batch_size=4, grad_accumulation=4)

    train_file = f"train_sentence_scale{args.scale}_seed{args.seed}.txt"
    if args.scale == "1.0":
        train_file = "train_sentence.txt"
    val_file = f"val_sentence.txt"   # should be a part of training data
    test_file = f"val_sentence.txt"     # currently, we use val.
    rel2template_file = "rel2info.txt"
    file_kwargs = dict(train_file=train_file,
                     val_file=val_file,
                     test_file=test_file,
                     rel2template_file=rel2template_file)
    data_dir = f"../DatasetForGeneration/{args.dataset}"
    gpt_size = "gpt2-large"
    output_dir = f"/data/jjyu/RE_Sentence_Generation/Fine_tuned_model/{args.dataset}/scale{args.scale}_seed{args.seed}/{gpt_size}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    PLM_path = f"/data/jjyu/PLMs/from_huggingface/{gpt_size}/"
    dir_kwargs = dict(
        data_dir=data_dir,
        output_dir=output_dir,
        model_dir=PLM_path
    )
    
    train_generator = TrainGenerator(dir_kwargs, file_kwargs, data_limit=limit)
    train_generator.train(model_kwargs, training_kwargs)

