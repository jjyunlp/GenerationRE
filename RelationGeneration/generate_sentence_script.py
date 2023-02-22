import os


cuda = 0
dataset = "re-tacred"
scale = "1.0"
seed = 0
prompt = "triplet"

cmd = f"CUDA_VISIBLE_DEVICES={cuda} python generate_sentence.py "\
	  f"--dataset {dataset} --scale {scale} --seed {seed} --prompt {prompt} "
      # f"--debug_num 10 "

print(cmd)
os.system(cmd)