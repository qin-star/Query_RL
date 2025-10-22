import json
import sys
import os
from tqdm import tqdm
import pdb
sys.path.append('./')

from src.Lucene.nfcorpus.search import PyseriniMultiFieldSearch
from src.Lucene.utils import ndcg_at_k
from src.eval.BM25.utils import parse_qrel


if not os.path.exists("data/local_index_search/nfcorpus/pyserini_index"):
    print("[Warning] Pyserini index not found for nfcorpus")
    search_system = None
else:
    search_system = PyseriniMultiFieldSearch(index_dir="data/local_index_search/nfcorpus/pyserini_index")


if __name__ == '__main__':
    with open("data/raw_data/nfcorpus/qrels/test.tsv", "r", encoding="utf-8") as file:
        qrel_test = [line.strip().split("\t") for line in file]

    qrel_test = qrel_test[1:]  # remove the header

    qrel_test = parse_qrel(qrel_test)

    # read code/data/raw_data/nfcorpus/queries.jsonl
    with open("data/raw_data/nfcorpus/queries.jsonl", "r", encoding="utf-8") as file:
        queries = [json.loads(line) for line in file]
    queries_dict = {q['_id']: q['text'] for q in queries}

    test_data = []
    for qid, value in qrel_test.items():
        test_data.append({
            "qid": qid,
            'query': queries_dict[qid],
            "target": value['targets'],
            "score": value['scores']
        })

    ndcg = []
    batch_size = 100

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [item['query'] for item in batch]
        targets = {item['query']: item['target'] for item in batch} 
        scores = {item['query']: item['score'] for item in batch}
        
        results = search_system.batch_search(queries, top_k=10, threads=16)
        
        for query in queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg.append(ndcg_at_k(retrieved, targets[query], 10, rel_scores=scores[query]))
    
    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")