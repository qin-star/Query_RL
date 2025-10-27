import json
import sys
import os
from tqdm import tqdm
import pdb
sys.path.append('./')

from pyserini.search.lucene import LuceneSearcher
from src.Lucene.utils import ndcg_at_k
from src.eval.BM25.utils import parse_qrel


if not os.path.exists("/shared/eng/pj20/lmr_model/raw_data/msmarco/indexes/lucene-index-msmarco-passage"):
    print("[Warning] Pyserini index not found for msmarco_beir")
    search_system = None
else:
    search_system = LuceneSearcher(index_dir="/shared/eng/pj20/lmr_model/raw_data/msmarco/indexes/lucene-index-msmarco-passage")


if __name__ == '__main__':
    # res_dir = '../results/Qwen-inst-msmarco_health.json'
    # res_dir = '../results/Qwen-inst-msmarco_science.json'
    # res_dir = '../results/Qwen-inst-msmarco_tech.json'
    # res_dir = '../results/no_reason/Qwen-inst-msmarco_health.json'
    # res_dir = '../results/no_reason/Qwen-inst-msmarco_science.json'
    # res_dir = '../results/no_reason/Qwen-inst-msmarco_tech.json'
    # res_dir = '../results/gpt-4o_post_msmarco_health.json'
    # res_dir = '../results/gpt-4o_post_msmarco_science.json'
    # res_dir = '../results/gpt-4o_post_msmarco_tech.json'
    # res_dir = '../results/claude-3.5_msmarco_health.json'
    # res_dir = '../results/claude-3.5_msmarco_science.json'
    # res_dir = '../results/claude-3.5_msmarco_tech.json'
    # res_dir = '../results/no_reason/gpt-4o_post_msmarco_health.json'
    # res_dir = '../results/no_reason/gpt-4o_post_msmarco_science.json'
    # res_dir = '../results/no_reason/gpt-4o_post_msmarco_tech.json'
    # res_dir = '../results/no_reason/claude-3.5_msmarco_health.json'
    # res_dir = '../results/no_reason/claude-3.5_msmarco_science.json'
    # res_dir = '../results/no_reason/claude-3.5_msmarco_tech.json'
    # res_dir = '../results/gpt-35_post_msmarco_health.json'
    # res_dir = '../results/gpt-35_post_msmarco_science.json'
    # res_dir = '../results/gpt-35_post_msmarco_tech.json'
    # res_dir = '../results/no_reason/gpt-35_post_msmarco_health.json'
    # res_dir = '../results/no_reason/gpt-35_post_msmarco_science.json'
    # res_dir = '../results/no_reason/gpt-35_post_msmarco_tech.json'
    # res_dir = '../results/no_reason/claude-haiku_post_msmarco_health.json'
    # res_dir = '../results/no_reason/claude-haiku_post_msmarco_science.json'
    # res_dir = '../results/no_reason/claude-haiku_post_msmarco_tech.json'
    # res_dir = '../results/claude-haiku_post_msmarco_health.json'
    # res_dir = '../results/claude-haiku_post_msmarco_science.json'
    res_dir = '../results/claude-haiku_post_msmarco_tech.json'

    with open(res_dir, "r", encoding="utf-8") as file:
        res = json.load(file)

    # convert the dict to a list
    test_data = [res[key] for key in res.keys()]
    
    ndcg = []

    for i in tqdm(range(0, len(test_data))):
        item = test_data[i]
        query = item['generated_text']
        target = eval(item['target'])
        scores = [1 for _ in range(len(target))]
        
        hits = search_system.search(query, k=10)
        doc_list = [json.loads(hit.lucene_document.get('raw'))['id'] for hit in hits]
        ndcg.append(ndcg_at_k(doc_list, target, 10, rel_scores=scores))

    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")