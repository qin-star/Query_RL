import sys
import os
import time
from functools import partial

# Debug: Print current directory and Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 
# Add the project root to Python path
sys.path.insert(0, project_root)  # 


import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import re
from collections import deque
from threading import Lock

from verl.utils.apis.ctgov import CTGovAPI


# Add these at the top with other global variables
_request_times = deque(maxlen=20)  # Track last 20 requests
_request_lock = Lock()  # Thread-safe lock for request tracking


def run_search_ctgov(search_query, search_api):
    # Rate limit checking
    current_time = time.time()
    with _request_lock:
        # Remove requests older than 1 second
        while _request_times and current_time - _request_times[0] > 1.0:
            _request_times.popleft()
        
        # Check if we're exceeding rate limit (10 requests per second)
        if len(_request_times) >= 10:
            print("\033[93m[Warning] CTGov rate limit (10 req/s) reached! Consider reducing batch size.\033[0m")
        
        # Record this request
        _request_times.append(current_time)
        
    nctid_list = search_api.search_with_query(search_query, topk=3000)
    return nctid_list

    
def extract_json_from_llm_output(text):
    extraction =  text.split("query\": ")[1].split("\n")[0].strip()
    if extraction.startswith("\"") and extraction.endswith("\""):
        return extraction[1:-1].replace("\"", "").replace("\\", "")
    else:
        return extraction



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="results/ctgov_reason")
    args = parser.parse_args()
    return args

def process_batch(generations, batch_data, search_api, llm):
    batch_recalls = []
    for idx, data in enumerate(batch_data):
        try:
            query = generations[idx]
            query = extract_json_from_llm_output(query)
            nctid_list = run_search_ctgov(query, search_api)
            gt_nctid_list = batch_data[idx]['trial_nctids']
            batch_recalls.append(len(set(nctid_list) & set(gt_nctid_list)) / len(gt_nctid_list))
        except Exception as e:
            print(f"Error in search query: {str(e)}")
            batch_recalls.append(0)
        
    return batch_recalls


def main():
    args = arg_parser()
    
    search_api = CTGovAPI()
        
    generations_path = "results/ctgov_reason/llm_responses_claude-hiku.json"
    test_path = "data/raw_data/ctgov/test.jsonl"
    test_data = []
    with open(test_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    with open(generations_path, 'r') as f:
        generations = json.load(f)
    
    recalls = []
    
    # Process data in batches
    batch_size = args.batch_size
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i + batch_size]
        generation = generations[i:i + batch_size]
        batch_recalls = process_batch(generation, batch, search_api, args.llm)
        recalls.extend(batch_recalls)
        
        current_avg_recall = sum(recalls) / len(recalls)
        print(f"Batch {i//batch_size + 1} completed. Current Average Recall: {current_avg_recall:.4f}")
    
    final_avg_recall = sum(recalls) / len(recalls)
    print(f"\nFinal Average Recall: {final_avg_recall:.4f}")
    
    # Save results

if __name__ == "__main__":
    main()
