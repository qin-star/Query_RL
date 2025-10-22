import json
import sys
import os
from tqdm import tqdm
import pdb
sys.path.append('./')

from src.Dense.scifact.search import FaissHNSWSearcher
from src.Lucene.utils import ndcg_at_k

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset to evaluate")
    parser.add_argument('--res_path', type=str, default="../results_dense/no_reason/gpt-4o_post_scifact.json", help="Path to the qrels file")
    args = parser.parse_args()
    
    # res_dir = '../results_dense/no_reason/gpt-35_post_scifact.json'
    # res_dir = '../results_dense/gpt-35_post_scifact.json'
    # res_dir = '../results_dense/no_reason/claude-haiku_post_scifact.json'
    res_dir = '../results_dense/claude-haiku_post_scifact.json'
    
    args.res_path = res_dir

    model_path = "intfloat/multilingual-e5-large-instruct"
    output_dir = "data/local_index_search/scifact/dense_index"
    index_path = f"{output_dir}/faiss_hnsw_index.bin"
    doc_ids_path = f"{output_dir}/doc_ids.npy"
    
    search_system = FaissHNSWSearcher(model_name=model_path, 
                                 index_path=index_path, 
                                 doc_ids_path=doc_ids_path)

    with open(args.res_path, "r", encoding="utf-8") as file:
        qrel_test = json.load(file)
    
    # transform qrel_test_dict to list
    test_data = []
    for qid, value in qrel_test.items():
        test_data.append(value)
    
    ndcg = []
    batch_size = 32

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [str(item['generated_text']) for item in batch]
        targets = {str(item['generated_text']): item['target'] for item in batch} 
        
        results = search_system.batch_search(queries, top_k=10, threads=16)
        for idx, query in enumerate(queries):
            retrieved = [result[0] for result in results[idx]]
            ndcg.append(ndcg_at_k(retrieved, targets[query], 10))
    
    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")