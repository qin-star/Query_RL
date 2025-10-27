import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from collections import defaultdict, Counter
import random
import pdb
import sqlite3




INSTRUCTION_SQL = """
You are a SQL query writing expert. Your task is to write the SQL query for the user query to retrieve data from a database.
"""




def generate_schema_prompt(db_id, split, num_rows=None):

    if split == 'train' or split == 'val':
        db_schema_path = f'data/raw_data/spider/spider_data/database/{db_id}/schema.sql'
    elif split == 'test':
        db_schema_path = f'data/raw_data/spider/spider_data/test_database/{db_id}/schema.sql'

    if not os.path.exists(db_schema_path):
        db_schema_path = db_schema_path.replace('schema', db_id)

    if not os.path.exists(db_schema_path):
        return ''

    schema_prompt = ''
    with open(db_schema_path, 'r') as f:
        for line in f:
            if 'insert' not in line.lower():
                schema_prompt += line

    return schema_prompt.strip()



def make_prefix(dp, split):
    
    instruction = INSTRUCTION_SQL

    input_str = """<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n""" + instruction

    # row example
    row_num = None
    # row_num = 3

    # table schema prompt
    schema_prompt = generate_schema_prompt(dp['db_id'], split, row_num)
    input_str += f"""Database Schema:
{schema_prompt}
"""


    input_str += """Note: Using valid SQLite and understanding External Knowledge, answer the following questions for the tables provided above.

Show your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer>. For example,
<think>
[thinking process]
</think>
<answer>
{
    "sql": "SELECT ... (in one line)"
} 
</answer>. 
"""

    input_str += """
Here's the user query:
"""

    input_str +=  dp['question'] + """<|im_end|>
<|im_start|>assistant
Let me write the SQL query with reasoning. 
<think>
"""
    return input_str


def load_bird_dataset():
    
    with open(f'data/raw_data/spider/spider_data/train_spider.json', 'r') as f:
        data_train_spider = json.load(f)

    with open(f'data/raw_data/spider/spider_data/train_others.json', 'r') as f:
        data_train_other = json.load(f)

    data_train = data_train_spider + data_train_other

    with open(f'data/raw_data/spider/spider_data/test.json', 'r') as f:
        data_test = json.load(f)
            
    with open(f'data/raw_data/spider/spider_data/dev.json', 'r') as f:
        data_val = json.load(f)
    
    train_data = [{'question': x['question'], 'db_id': x['db_id'], 'sql': x['query']} for x in data_train]
    test_data = [{'question': x['question'], 'db_id': x['db_id'], 'sql': x['query']} for x in data_test]
    val_data = [{'question': x['question'], 'db_id': x['db_id'], 'sql': x['query']} for x in data_val]
    
    return train_data, test_data, val_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/sql')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--dataset', type=str, default='spider')
    parser.add_argument('--output_dir', type=str, default='data/sql')

    args = parser.parse_args()
    
    data_source = args.dataset
    

    train_data, test_data, val_data = load_bird_dataset()

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    val_dataset = Dataset.from_list(val_data)
    
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, split)
            solution = {
                "target": example['sql'],
            }

            db_id = example['db_id']
            if split == 'train' or split == 'val':
                db_path = f'data/raw_data/spider/spider_data/database/{db_id}/{db_id}.sqlite'
            elif split == 'test':
                db_path = f'data/raw_data/spider/spider_data/test_database/{db_id}/{db_id}.sqlite'

            data = {
                "data_source": f"{data_source}_{split}",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "sql_generation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'db_path': db_path
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True)
    # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)
    

    print(f"Train dataset size before filtering: {train_dataset.num_rows}")
    train_dataset = train_dataset.filter(
        lambda example: len(example['prompt'][0]['content'].split()) <= 1000
    )
    print(f"Train dataset size after filtering: {train_dataset.num_rows}")

    lengths_list = []
    for d in train_dataset:
        lengths_list.append(len(d['prompt'][0]['content'].split()))

    lengths_list_test = []
    for d in test_dataset:
        lengths_list_test.append(len(d['prompt'][0]['content'].split()))
        
    lengths_list_val = []
    for d in val_dataset:
        lengths_list_val.append(len(d['prompt'][0]['content'].split()))
        
    print(f"Average length of train dataset: {sum(lengths_list) / len(lengths_list)}")
    print(f"Average length of test dataset: {sum(lengths_list_test) / len(lengths_list_test)}")
    print(f"Average length of val dataset: {sum(lengths_list_val) / len(lengths_list_val)}")
    
    print(f"Max length of train dataset: {max(lengths_list)}")
    print(f"Max length of test dataset: {max(lengths_list_test)}")
    print(f"Max length of val dataset: {max(lengths_list_val)}")

    local_dir = os.path.join(args.local_dir, f"{args.dataset}")
    hdfs_dir = os.path.join(args.hdfs_dir, f"{args.dataset}") if args.hdfs_dir is not None else None
    
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 