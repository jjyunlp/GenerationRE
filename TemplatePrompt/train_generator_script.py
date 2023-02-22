import os

cuda = 0
dataset = "re-tacred_0.1"
dataset = "re-tacred"
scale = "1.0"
seed = 0
prompt = "template"
prompt = "templateV2"
prompt = "templateV3"

cmd = f"CUDA_VISIBLE_DEVICES={cuda} python train_generator.py "\
	  f"--dataset {dataset} --scale {scale} --seed {seed} --prompt {prompt}"

print(cmd)
os.system(cmd)

cmd = f"CUDA_VISIBLE_DEVICES={cuda} python generate_sentence.py "\
	  f"--dataset {dataset} --scale {scale} --seed {seed} --prompt {prompt} "

print(cmd)
os.system(cmd)
