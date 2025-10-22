import sys
import os

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
import time

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
        
    nctid_list = search_api.search_with_keywords(search_query, topk=3000)
    return nctid_list
    

def main():
    search_api = CTGovAPI()
        
    test_path = "data/raw_data/ctgov/test.jsonl"
    test_data = []
    with open(test_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    recalls = []
    for data in tqdm(test_data):
        try:
            query = f"{data['pico']['P']} OR {data['pico']['I']} OR {data['pico']['C']} OR {data['pico']['O']}"
            ground_truth = data['trial_nctids']
            
            nctid_list = run_search_ctgov(query, search_api)
            
            recall = len(set(nctid_list) & set(ground_truth)) / len(ground_truth)
            recalls.append(recall)
        
            print(f"Average Recall: {sum(recalls) / len(recalls)}")
        except Exception as e:
            print(f"Error: {e}")
            recalls.append(0)
            continue
        

if __name__ == "__main__":
    main()
