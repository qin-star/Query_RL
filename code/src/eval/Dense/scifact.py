import json
import sys
import os
from tqdm import tqdm
import pdb
sys.path.append('./')

from src.Dense.scifact.search import FaissHNSWSearcher
from src.Lucene.utils import ndcg_at_k


if not os.path.exists("data/local_index_search/scifact/dense_index"):
    print("[Warning] Faiss index not found for scifact")
    search_system = None
else:
    model_path = "intfloat/multilingual-e5-large-instruct"
    output_dir = "data/local_index_search/scifact/dense_index"
    index_path = f"{output_dir}/faiss_hnsw_index.bin"
    doc_ids_path = f"{output_dir}/doc_ids.npy"
    
    search_system = FaissHNSWSearcher(model_name=model_path, 
                                 index_path=index_path, 
                                 doc_ids_path=doc_ids_path)



if __name__ == '__main__':
    with open("data/raw_data/scifact/qrels/test.tsv", "r", encoding="utf-8") as file:
        qrel_test = [line.strip().split("\t") for line in file]

    qrel_test = qrel_test[1:]  # remove the header

    # read code/data/raw_data/scifact/queries.jsonl
    with open("data/raw_data/scifact/queries.jsonl", "r", encoding="utf-8") as file:
        queries = [json.loads(line) for line in file]
    queries_dict = {q['_id']: q['text'] for q in queries}

    test_data = []
    for qid, docid, label in qrel_test:
        test_data.append({
            "qid": qid,
            'query': queries_dict[qid],
            "target": docid,
            "label": int(label)
        })


    ndcg = []
    batch_size = 32

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [item['query'] for item in batch]
        targets = {item['query']: item['target'] for item in batch}
        
        results = search_system.batch_search(queries, top_k=10, threads=16)
        for idx, query in enumerate(queries):
            retrieved = [result[0] for result in results[idx]]
            ndcg.append(ndcg_at_k(retrieved, targets[query], 10))
    
    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")