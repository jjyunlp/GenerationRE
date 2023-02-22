import os

dataset = "re-tacred_1_9"
# dataset = "re-tacred"
scale = "1.0"
seed = 0
gpt = "gpt2-large"
prompt = "template"
prompt = "templateV2"
prompt = "template"
prompt = "triplet"

cmd = f"python build_supervised_dataset_by_generated_sentence.py "\
	  f"--dataset {dataset} --scale {scale} --seed {seed} --gpt {gpt} --prompt {prompt}"

print(cmd)
os.system(cmd)
