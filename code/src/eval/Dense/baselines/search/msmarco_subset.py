import json
import sys
import os
from tqdm import tqdm
import pdb
import torch
sys.path.append('./')

from src.Lucene.utils import ndcg_at_k
from src.eval.BM25.utils import parse_qrel

from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder

class PyseriniFaissSearcher:
    def __init__(self, index_dir, model_name):
        """
        Initialize the Pyserini FAISS searcher.
        :param index_dir: Path to the FAISS index directory.
        :param model_name: Name of the transformer model for query encoding.
        """
        self.index_dir = index_dir
        self.model_name = model_name
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_encoder = AutoQueryEncoder(model_name, pooling='mean', device=device)
        self.searcher = FaissSearcher(index_dir=index_dir, query_encoder=query_encoder)
    
    def search(self, query, top_k=5, threads=8):
        """ Perform a search for a single query. """
        hits = self.searcher.search(query, k=top_k, threads=threads)
        results = [(hit.docid, hit.score) for hit in hits]
        return results
    
    def batch_search(self, queries, top_k=5, threads=8):
        """ Perform batch search for multiple queries. """
        batch_results = self.searcher.batch_search(queries, queries, k=top_k, threads=threads)
        results = {query: [(hit.docid, hit.score) for hit in batch_results[query]] for query in queries}
        return results


index_path = '/shared/eng/pj20/lmr_model/raw_data/msmarco/indexes/contriever-msmarco-passage-dense-index'
model_name = 'facebook/contriever'

if not os.path.exists(index_path):
    print("[Warning] Pyserini index not found for msmarco_beir")
    search_system = None
else:
    search_system = PyseriniFaissSearcher(index_dir=index_path, model_name=model_name)


if __name__ == '__main__':
    # res_dir = '../results_dense/Qwen-inst-msmarco_health.json'
    # res_dir = '../results_dense/Qwen-inst-msmarco_science.json'
    # res_dir = '../results_dense/Qwen-inst-msmarco_tech.json'
    # res_dir = '../results_dense/no_reason/Qwen-inst-msmarco_health.json'
    # res_dir = '../results_dense/no_reason/Qwen-inst-msmarco_science.json'
    # res_dir = '../results_dense/no_reason/Qwen-inst-msmarco_tech.json'
    # res_dir = '../results_dense/gpt-4o_post_msmarco_health.json'
    # res_dir = '../results_dense/gpt-4o_post_msmarco_science.json'
    # res_dir = '../results_dense/gpt-4o_post_msmarco_tech.json'
    # res_dir = '../results_dense/claude-3.5_post_msmarco_health.json'
    # res_dir = '../results_dense/claude-3.5_post_msmarco_science.json'
    # res_dir = '../results_dense/claude-3.5_post_msmarco_tech.json'
    # res_dir = '../results_dense/no_reason/gpt-4o_post_msmarco_health.json'
    # res_dir = '../results_dense/no_reason/gpt-4o_post_msmarco_science.json'
    # res_dir = '../results_dense/no_reason/gpt-4o_post_msmarco_tech.json'
    # res_dir = '../results_dense/no_reason/claude-3.5_post_msmarco_health.json'
    # res_dir = '../results_dense/no_reason/claude-3.5_post_msmarco_science.json'
    # res_dir = '../results_dense/no_reason/claude-3.5_post_msmarco_tech.json'

    # res_dir = '../results_dense/no_reason/claude-haiku_post_msmarco_health.json'
    # res_dir = '../results_dense/no_reason/claude-haiku_post_msmarco_science.json'
    # res_dir = '../results_dense/no_reason/claude-haiku_post_msmarco_tech.json'
    # res_dir = '../results_dense/claude-haiku_post_msmarco_health.json'
    # res_dir = '../results_dense/claude-haiku_post_msmarco_science.json'
    # res_dir = '../results_dense/claude-haiku_post_msmarco_tech.json'
    # res_dir = '../results_dense/gpt-35_post_msmarco_health.json'
    # res_dir = '../results_dense/gpt-35_post_msmarco_science.json'
    # res_dir = '../results_dense/gpt-35_post_msmarco_tech.json'
    # res_dir = '../results_dense/no_reason/gpt-35_post_msmarco_health.json'
    # res_dir = '../results_dense/no_reason/gpt-35_post_msmarco_science.json'
    res_dir = '../results_dense/no_reason/gpt-35_post_msmarco_tech.json'

    print(f"Reading {res_dir}")
    with open(res_dir, "r", encoding="utf-8") as file:
        res = json.load(file)

    # convert the dict to a list
    test_data = [res[key] for key in res.keys()]
    
    ndcg = []

    for i in tqdm(range(0, len(test_data))):
        item = test_data[i]
        query = str(item['generated_text'])
        target = eval(item['target'])
        scores = [1 for _ in range(len(target))]
        
        results = search_system.batch_search([query], top_k=10, threads=16)
        retrieved = [result[0] for result in results.get(query, [])]
        ndcg.append(ndcg_at_k(retrieved, target, 10, rel_scores=scores))

    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")