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
import records
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sql.wikisql_lib.dbengine import DBEngine
from src.sql.wikisql_lib.query import Query, SQLToJsonConverter
from src.sql.wikisql_lib.common import count_lines


INSTRUCTION_SQL = """
You are a SQL query writing expert. Your task is to write the SQL query for the user query to retrieve data from a database.
"""



# def generate_schema_prompt(db_id, split, num_rows=None):

#     if split == 'train' or split == 'test':
#         table_schema_path = f'data/raw_data/wikisql/WikiSQL/data/test.tables.jsonl'
#     elif split == 'val':
#         table_schema_path = f'data/raw_data/wikisql/WikiSQL/data/dev.tables.jsonl'

#     table_schema = None
#     with open(table_schema_path, 'r') as f:
#         for line in f:
#             D = json.loads(line)
#             if D['id'] == db_id:
#                 table_schema = D
#                 break

#     assert table_schema is not None, f"Table schema not found for db_id: {db_id}"

#     schema_prompt = ""
#     if 'page_title' in table_schema:
#         schema_prompt += f"Page Title: {table_schema['page_title']}\n"
#     if 'section_title' in table_schema:
#         schema_prompt += f"Section Title: {table_schema['section_title']}\n"
#     if 'caption' in table_schema:
#         schema_prompt += f"Caption: {table_schema['caption']}\n"
#     schema_prompt += f"Header: {table_schema['header']}\n"
#     schema_prompt += f"Column Types: {table_schema['types']}\n"

#     if num_rows is not None:
#         schema_prompt += f"""Rows: {table_schema['rows']}"""

#     return schema_prompt



def generate_schema_prompt(db_id, split, num_rows=None):

    if split == 'train' or split == 'test':
        table_path = f'data/raw_data/wikisql/WikiSQL/data/{split}.db'
    elif split == 'val':
        table_path = f'data/raw_data/wikisql/WikiSQL/data/dev.db'

    db = records.Database('sqlite:///{}'.format(table_path))
    conn = db.get_connection()

    if 'table' in db_id:
        table_name = db_id
    else:
        table_name = 'table_' + db_id.replace('-', '_')

    print(table_name)
    schema_prompt = conn.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_name).all()[0].sql
    return schema_prompt




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

    input_str += """Note: Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.

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

#     input_str += """
# Important rules:  
# - You should only think once and answer once.
# - You should only include one <think> </think> block and one <answer> </answer> block.  
# - End your response immediately after the <answer> </answer> tag — no extra text.  
# """

    input_str += """
Here's the user query:
"""

    input_str +=  dp['question'] + """<|im_end|>
<|im_start|>assistant
Let me write the SQL query with reasoning. 
<think>
"""
    return input_str


def load_wikisql_dataset():
    
    data_train = []
    with open(f'data/raw_data/wikisql/WikiSQL/data/train.jsonl', 'r') as f:
        for line in f:
            data_train.append(json.loads(line))

    data_test = []
    with open(f'data/raw_data/wikisql/WikiSQL/data/test.jsonl', 'r') as f:
        for line in f:
            data_test.append(json.loads(line))
            
    data_val = []
    with open(f'data/raw_data/wikisql/WikiSQL/data/dev.jsonl', 'r') as f:
        for line in f:
            data_val.append(json.loads(line))
    data_val = data_val[:100]

    def process_wikisql_data(data):
        processed_data = []
        for x in data:
            sql = Query.from_dict(x['sql'])
            sql = sql.to_query(['text'])
            # sql = x['sql']
            print(sql)
            processed_data.append({
                'question': x['question'],
                'db_id': x['table_id'],
                'sql': sql
            })
        return processed_data
    
    train_data = process_wikisql_data(data_train)
    test_data = process_wikisql_data(data_test)
    val_data = process_wikisql_data(data_val)
    
    return train_data, test_data, val_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/wikisql')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--dataset', type=str, default='wikisql')
    parser.add_argument('--output_dir', type=str, default='data/wikisql')

    args = parser.parse_args()
    
    data_source = args.dataset
    

    train_data, test_data, val_data = load_wikisql_dataset()

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
            db_path = f'data/raw_data/wikisql/WikiSQL/data/{split}.db'

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
                    'db_path': db_path,
                    'db_id': db_id
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
    
    # print(f"Train dataset size before filtering: {train_dataset.num_rows}")
    # train_dataset = train_dataset.filter(
    #     lambda example: len(example['prompt'][0]['content'].split()) <= 1000
    # )
    # print(f"Train dataset size after filtering: {train_dataset.num_rows}")

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