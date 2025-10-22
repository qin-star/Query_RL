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



INSTRUCTION_SPARSE = """
You are a query rewriting expert. Your task is to augment the user query to find relevant literature in a corpus with sparse retrieval.
"""

INSTRUCTION_DENSE = """
You are a query rewriting expert. Your task is to augment the user query to find relevant literature in a corpus with dense retrieval.
"""


def make_prefix(dp, retrieval_mode):
    
    if retrieval_mode == 'sparse':
        instruction = INSTRUCTION_SPARSE
    elif retrieval_mode == 'dense':
        instruction = INSTRUCTION_DENSE
    else:
        raise ValueError(f"Invalid retrieval mode: {retrieval_mode}")

    input_str = """<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n""" + instruction
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<think>
[thinking process]
</think>
<answer>
{
    "query": "...."
} 
</answer>. 
"""
    if retrieval_mode == 'sparse':
        input_str += """Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately."
"""
    elif retrieval_mode == 'dense':
        input_str += """Note: Given a question, your need to retain the original query while expanding it with additional semantically relevant information, to retrieve documents that can help answer the question. If no useful expansion is needed, return the original query as is.
"""

    input_str += """
Here's the user query:
"""

    input_str +=  dp['input'] + """<|im_end|>
<|im_start|>assistant\n
Let me rewrite the query with reasoning. 
<think>
"""

    return input_str


def load_matching_dataset(domain):
    
    data_train = []
    data_test = []
    data_val = []
    
    with open(f'data/raw_data/msmarco/msmarco_{domain}/train.jsonl', 'r') as f:
        for line in f:
            data_train.append(json.loads(line))

    with open(f'data/raw_data/msmarco/msmarco_{domain}/dev.jsonl', 'r') as f:
        for line in f:
            data_test.append(json.loads(line))
            
    with open(f'data/raw_data/msmarco/msmarco_{domain}/dev.jsonl', 'r') as f:
        cnt = 0
        for line in f:
            data_val.append(json.loads(line))
            cnt += 1
            if cnt > 100:
                break
    
    train_data = [{'input': x['question'], 'label': x['docs_id']} for x in data_train]
    test_data = [{'input': x['question'], 'label': x['docs_id']} for x in data_test]
    val_data = [{'input': x['question'], 'label': x['docs_id']} for x in data_val]
    
    return train_data, test_data, val_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/local_index_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--dataset', type=str, default='msmarco')
    parser.add_argument('--domains', type=list, default=['all', 'health', 'science', 'tech'])
    parser.add_argument('--output_dir', type=str, default='data/local_index_search')

    args = parser.parse_args()
    
    data_source = args.dataset
    
    for retrieval_mode in ['sparse', 'dense']:
        for domain in args.domains:
            train_data, test_data, val_data = load_matching_dataset(domain)

            train_dataset = Dataset.from_list(train_data)
            test_dataset = Dataset.from_list(test_data)
            val_dataset = Dataset.from_list(val_data)
            
            def make_map_fn(split):
                def process_fn(example, idx):
                    question = make_prefix(example, retrieval_mode)
                    solution = {
                        "target": example['label'],
                    }
                    data = {
                        "data_source": f"{data_source}_{retrieval_mode}_{split}",
                        "prompt": [{
                            "role": "user",
                            "content": question,
                        }],
                        "ability": "literature_mining",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": solution
                        },
                        "extra_info": {
                            'split': split,
                            'index': idx,
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
            
            local_dir = os.path.join(args.local_dir, f"{args.dataset}_{domain}", retrieval_mode)
            hdfs_dir = os.path.join(args.hdfs_dir, f"{args.dataset}_{domain}", retrieval_mode) if args.hdfs_dir is not None else None
            
            os.makedirs(local_dir, exist_ok=True)
            
            train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
            test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
            val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))
            
            if hdfs_dir is not None:
                makedirs(hdfs_dir)
                copy(src=local_dir, dst=hdfs_dir) 