import json
import logging
import os
from typing import List

def load_data_from_file(input_file: str) -> List:
	assert os.path.isfile(input_file), f"{input_file} not exist."
	data = []
	with open(input_file) as reader:
		for line in reader.readlines():
			data.append(json.loads(line))
	logging.info(f"Load {len(data)} instances from {input_file}.")		
	return data

def dump_data_to_file(output_file: str, data: List, overwrite=False) -> None:
	if os.path.isfile(output_file) and not overwrite:
		print("File exist and not overwrite")
		exit()
	with open(output_file, 'w') as writer:
		for inst in data:
			writer.write(json.dumps(inst) + '\n')


def load_labels_from_file(input_file: str)-> dict:
	"""Directly load all relaitons to id in a dict json file

	Args:
		input_file (str): _description_

	Returns:
		dict: _description_
	"""
	with open(input_file, 'r') as reader:
		rel2id = json.load(reader)
	return rel2id

def dump_labels_to_file(data, output_file: str, overwrite=False)-> None:
	"""Directly dump all label mapping info in a dict json file

	Args:
		input_file (str): _description_

	Returns:
		dict: _description_
	"""
	with open(output_file, 'w') as writer:
		json.dump(data, writer, indent=4)