import os

cuda = 0
dataset = "re-tacred_0.1"
scale = "1.0"
seed = 0
prompt = "triplet"

cmd = f"CUDA_VISIBLE_DEVICES={cuda} python train_generator.py "\
	  f"--dataset {dataset} --scale {scale} --seed {seed} --prompt {prompt}"

print(cmd)
os.system(cmd)
