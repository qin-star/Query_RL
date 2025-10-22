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

from verl.utils.apis.pubmed import PubmedAPI

# Add these at the top with other global variables
_request_times = deque(maxlen=20)  # Track last 20 requests
_request_lock = Lock()  # Thread-safe lock for request tracking



def run_search_pubmed(search_query, search_api, pub_date):
    # Rate limit checking
    current_time = time.time()
    with _request_lock:
        # Remove requests older than 1 second
        while _request_times and current_time - _request_times[0] > 1.0:
            _request_times.popleft()
        
        # Check if we're exceeding rate limit (10 requests per second)
        if len(_request_times) >= 10:
            print("\033[93m[Warning] PubMed rate limit (10 req/s) reached! Consider reducing batch size.\033[0m")
        
        # Record this request
        _request_times.append(current_time)
    
    # add date
    date_query_part = f'&datetype=pdat&mindate=1970/01/01&maxdate={pub_date}'
    search_query += date_query_part
    
    print('Query:', search_query)
    # search
    pmid_list = search_api.search_with_keywords(search_query, topk=3000)
    
    return pmid_list
    

def main():
    if os.path.exists('verl/utils/reward_score/apis/pubmed_api.key'):
        api_key = open('verl/utils/reward_score/apis/pubmed_api.key', 'r').read().strip()
        search_api = PubmedAPI(api_key=api_key)
        
    test_path = "data/raw_data/pubmed/test.jsonl"
    test_data = []
    with open(test_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    recalls = []
    results = {}
    cnt = 0
    for data in tqdm(test_data):
        query = f"{data['pico']['P']} OR {data['pico']['I']} OR {data['pico']['C']} OR {data['pico']['O']}"
        pub_date = data['pub_date']
        ground_truth = data['publication_pmids']
        
        ranks = []
        
        pmid_list = run_search_pubmed(query, search_api, pub_date)
        
        recall = len(set(pmid_list) & set(ground_truth)) / len(ground_truth)
        recalls.append(recall)
        
        for gt in ground_truth:
            if gt in pmid_list:
                ranks.append(pmid_list.index(gt))
            else:
                ranks.append(999999)
        
        results[cnt] = {
            'query': query,
            'ground_truth': ground_truth,
            # 'pmid_list': pmid_list,
            'ranks': ranks,
            'recall': recall
        }
        cnt += 1
    
        print(f"Average Recall: {sum(recalls) / len(recalls)}")
        
        # save results
        if cnt % 100 == 0:
            with open('results/pubmed_base_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        

if __name__ == "__main__":
    main()
