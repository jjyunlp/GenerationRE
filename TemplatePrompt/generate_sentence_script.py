import os


cuda = 1
dataset = "re-tacred_0.1"
scale = "1.0"
seed = 0
prompt = "templateV3"

cmd = f"CUDA_VISIBLE_DEVICES={cuda} python generate_sentence.py "\
	  f"--dataset {dataset} --scale {scale} --seed {seed} --prompt {prompt} "

print(cmd)
os.system(cmd)