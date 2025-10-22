import json
import sys
import os
from tqdm import tqdm
import pdb
sys.path.append('./')

from src.Lucene.scifact.search import PyseriniMultiFieldSearch
from src.Lucene.utils import ndcg_at_k

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset to evaluate")
    args = parser.parse_args()

    # res_path = '../results/claude-3.5_post_scifact.json'
    # res_path = '../results/no_reason/Qwen-inst-scifact.json'
    # res_path = '../results/no_reason/claude-3.5_post_scifact.json'
    # res_path = '../results/no_reason/gpt-4o_post_scifact.json'
    # res_path = '../results/no_reason/gpt-35_post_scifact.json'
    # res_path = '../results/gpt-35_post_scifact.json'
    # res_path = '../results/no_reason/claude-haiku_post_scifact.json'
    res_path = '../results/claude-haiku_post_scifact.json'
    
    search_system = PyseriniMultiFieldSearch(index_dir=f"data/local_index_search/{args.dataset}/pyserini_index")

    with open(res_path, "r", encoding="utf-8") as file:
        qrel_test = json.load(file)
    
    # transform qrel_test_dict to list
    test_data = []
    for qid, value in qrel_test.items():
        test_data.append(value)

    ndcg = []
    batch_size = 100

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [str(item['generated_text']) for item in batch]
        targets = {str(item['generated_text']): item['target'] for item in batch} 
        
        results = search_system.batch_search(queries, top_k=10, threads=16)
        
        for query in queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg.append(ndcg_at_k(retrieved, targets[query], 10))
    
    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")