import json
import sys
import os
from tqdm import tqdm
import pdb
sys.path.append('./')

from src.Lucene.fever.search import PyseriniMultiFieldSearch
from src.Lucene.utils import ndcg_at_k
from src.eval.BM25.utils import parse_qrel


if not os.path.exists("data/local_index_search/fever/pyserini_index"):
    print("[Warning] Pyserini index not found for fever")
    search_system = None
else:
    search_system = PyseriniMultiFieldSearch(index_dir="data/local_index_search/fever/pyserini_index")


if __name__ == '__main__':
    # res_dir = '../results/Qwen-inst-fever.json'
    # res_dir = '../results/gpt-4o_post_fever.json'
    # res_dir = '../results/claude-3.5_post_fever.json'
    # res_dir = '../results/no_reason/Qwen-inst-fever.json'
    # res_dir = '../results/no_reason/claude-3.5_post_fever.json'
    # res_dir = '../results/no_reason/gpt-4o_post_fever.json'
    
    # res_dir = '../results/no_reason/gpt-35_post_fever.json'
    # res_dir = '../results/gpt-35_post_fever.json'
    # res_dir = '../results/no_reason/claude-haiku_post_fever.json'
    res_dir = '../results/claude-haiku_post_fever.json'

    with open(res_dir, "r", encoding="utf-8") as file:
        res = json.load(file)
    
    with open("data/raw_data/fever/qrels/test.tsv", "r", encoding="utf-8") as file:
        qrel_test = [line.strip().split("\t") for line in file]

    qrel_test = qrel_test[1:]  # remove the header

    qrel_test = parse_qrel(qrel_test)

    # read code/data/raw_data/fever/queries.jsonl
    with open("data/raw_data/fever/queries.jsonl", "r", encoding="utf-8") as file:
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
        queries = [str(res[item['qid']]['generated_text']) for item in batch]
        targets = {str(res[item['qid']]['generated_text']): item['target'] for item in batch} 
        scores = {str(res[item['qid']]['generated_text']): item['score'] for item in batch}
        
        results = search_system.batch_search(queries, top_k=10, threads=16)
        
        for query in queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg.append(ndcg_at_k(retrieved, targets[query], 10, rel_scores=scores[query]))
    
    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")