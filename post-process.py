"""

"""

from data_load_and_dump import load_data_from_file, dump_data_to_file


class UpdateNA():
	"""After generation with pseudo-relation for NA instances, 
	we will check if the generated head and tail entites are exactly the original entity pair.
	Input two files: 1) the origial instance file; 2) the output of GPT2
	"""
	def __init__(self) -> None:
		pass


def compare(a: str, b: str)->bool:
	"""Check whether a is the same with b or it is a substring of b 

	Args:
		a (str): the generate entity
		b (str): the original entity

	Returns:
		bool: _description_
	"""
	a = a.lower()
	b = b.lower()
	if a == b:
		return True
	if a in b:	# substring ? maybe more strict
		return True
	return False

def update_na_data(input_file: str, output_file: str) -> list:
	"""Update the NA instances in distantly supervision data

	Args:
		input_file (str): _description_
		output_file (str): _description_

	Returns:
		list: _description_
	"""
	ds_data = load_data_from_file(input_file)
	gpt_data = load_data_from_file(output_file)

	# gpt的结果有多种形式，有的有多个pred，有的没有预测。都需要处理
	ds_inst_with_gpt_pred = []
	current_index = 0
	for ds_inst in ds_data:
		ds_sen = ds_inst['text']
		gpt_inst = gpt_data[current_index]
		# gpt_sen = gpt_inst['text']
		text_a = gpt_inst['text_a']
		prompt = text_a.split('\nSentence: ')[-1]
		gpt_sen, rel_and_head = prompt.split("\nRelation: ")
		gpt_sen = gpt_sen[1:-1]		# delete ""
		head = "None"
		tail = "None"
		gpt_rel = "None"
		if ds_sen == gpt_sen:
			gpt_rel = rel_and_head.split("\nHead Entity: ")[0]
			gpt_rel = gpt_rel[1:-1]		# 这个relation被转换成容易识别的，比如person/person -> person / person
			gpt_rel = gpt_rel.replace(" / ", "/")
			gpt_rel = gpt_rel.replace(" ", "_")
			# To extract head and tail entity
			text_b = gpt_inst['text_b']
			text_b_items = text_b.split("\"")
			if len(text_b_items) >= 3:
				# at least 3 items
				head, _, tail = text_b_items[:3]
			# some ds_inst will keep unchanged if they dont appear in gpt
			current_index += 1
		# no matter how, we will add the ds inst
		ds_inst['gpt_head'] = head
		ds_inst['gpt_tail'] = tail
		ds_inst['gpt_relation'] = gpt_rel
		ds_inst_with_gpt_pred.append(ds_inst)
	print(ds_inst_with_gpt_pred[:10])
	print(len(ds_inst_with_gpt_pred))

	updated_data = []
	unchanged_data = []
	for inst in ds_inst_with_gpt_pred:
		ds_h = inst['h']['name']
		ds_t = inst['t']['name']
		gpt_h = inst['gpt_head']
		gpt_t = inst['gpt_tail']
		if compare(gpt_h, ds_h) and compare(gpt_t, ds_t):
			inst['relation'] = inst['gpt_relation']
			updated_data.append(inst)
		else:
			unchanged_data.append(inst)
	
	print(len(updated_data))

	return (updated_data, unchanged_data)



if __name__ == "__main__":
	na_file = "/home/jjy/work/OpenNRE/benchmark/nyt10m/fewshot_v2/nyt10m_na_ds_train.txt"
	gpt_file = "/home/jjy/work/GenerateDataForRE/dino/output/nyt10m_gpt2-xl_relation2entity_task_5_42_n1_rel2entity_split_rel_for_na_contains/relation2entity_gpt-dataset.jsonl"
	nonna_data, na_data = update_na_data(na_file, gpt_file)
	filtered_na_file = "/home/jjy/work/OpenNRE/benchmark/nyt10m/fewshot_v2/nyt10m_filtered_na_ds_train.txt"
	new_nonna_file = "/home/jjy/work/OpenNRE/benchmark/nyt10m/fewshot_v2/nyt10m_new_nonna_ds_train_rel-contain.txt"
	dump_data_to_file(filtered_na_file, na_data)
	dump_data_to_file(new_nonna_file, nonna_data)
	





