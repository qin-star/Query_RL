import re
import os
from datasets import Dataset, load_dataset, concatenate_datasets
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from collections import defaultdict, Counter
import random
import pdb



INSTRUCTION = """
You are an expert in query generation. Given a question, your task is to create query terms to retrieve relevant documents that best answer the question."""


def make_prefix(dp):

    input_str = """<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n""" + INSTRUCTION
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<answer>
{
    "query": "...."
} 
</answer>. 
Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately.

Here's the question:
"""
    input_str +=  dp['query'] + """
Assistant: Let me think step by step. 
<think>
"""

    return input_str



def parse_qrel(qrel):
    """
    Parse a TSV file and return a dictionary where:
    - Keys are query IDs.
    - Values are dictionaries containing 'targets' (list of corpus IDs) and 'scores' (list of scores).
    """
    query_dict = {}

    # Skip the header
    for line in qrel:
        query_id, corpus_id, score = line

        if query_id not in query_dict:
            query_dict[query_id] = {"targets": [], "scores": []}

        query_dict[query_id]["targets"].append(corpus_id)
        query_dict[query_id]["scores"].append(int(score))

    return query_dict


def load_matching_dataset():
    # code/data/raw_data/nfcorpus/qrels/*.tsv
    with open("code/data/raw_data/nfcorpus/qrels/train.tsv", "r", encoding="utf-8") as file:
        qrel_train = [line.strip().split("\t") for line in file]
    
    qrel_train = qrel_train[1:]  # remove the header
    
    with open("code/data/raw_data/nfcorpus/qrels/test.tsv", "r", encoding="utf-8") as file:
        qrel_test = [line.strip().split("\t") for line in file]

    qrel_test = qrel_test[1:]  # remove the header

    with open("code/data/raw_data/nfcorpus/qrels/dev.tsv", "r", encoding="utf-8") as file:
        qrel_val = [line.strip().split("\t") for line in file]

    qrel_val = qrel_val[1:]  # remove the header

    qrel_train = parse_qrel(qrel_train)
    qrel_test = parse_qrel(qrel_test)
    qrel_val = parse_qrel(qrel_val)


    # read code/data/raw_data/nfcorpus/queries.jsonl
    with open("code/data/raw_data/nfcorpus/queries.jsonl", "r", encoding="utf-8") as file:
        queries = [json.loads(line) for line in file]

    # transform the queries into a dictionary
    queries_dict = {q['_id']: q['text'] for q in queries}
    # process the data
    def process_qrel(qrel):
        data = []
        for qid, value in qrel.items():
            data.append({
                "qid": qid,
                'query': queries_dict[qid],
                "target": value['targets'],
                "score": value['scores']
            })
        return data

    train_data = process_qrel(qrel_train)
    test_data = process_qrel(qrel_test)
    val_data = process_qrel(qrel_val)
    
    return train_data, test_data, val_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='code/data/local_index_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--dataset', type=str, default='nfcorpus')

    args = parser.parse_args()
    
    data_source = args.dataset
    
    train_data, test_data, val_data = load_matching_dataset()


    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    val_dataset = Dataset.from_list(val_data)


    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example)
            solution = {
                "target": example['target'],
                'score': example['score']
            }
            data = {
                "data_source": data_source + '_' + split,
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

    val_dataset = concatenate_datasets([val_dataset, test_dataset])
    
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
    
    local_dir = os.path.join(args.local_dir, args.dataset)
    hdfs_dir = os.path.join(args.hdfs_dir, args.dataset) if args.hdfs_dir is not None else None
    
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 