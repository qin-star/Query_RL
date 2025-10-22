import re
import random
import os
import json
try:
    import utils.java_init
except:
    print("Failed to import java_init")
    pass

from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from tqdm import tqdm


_searcher = None
_tokenizer = SimpleTokenizer()

def get_searcher():
    """Lazily initialize and return the searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
    return _searcher

def run_index_search_bm25(search_query, topk=50):
    
    searcher = get_searcher()
    
    # Rate limit checking
    hits = searcher.search(search_query, k=topk)
    
    doc_list = [json.loads(hit.lucene_document.get('raw'))['contents'] for hit in hits]
    
    return doc_list
    

def main():
        
    test_path = "data/raw_data/squad/test.jsonl"
    test_data = []
    with open(test_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    recall_at_20 = []
    recall_at_50 = []
    recall_at_100 = []
    recall_at_1000 = []
    
    
    
    for data in tqdm(test_data):
        query = data['question']
        ground_truth = data['answers']
        
        doc_list = run_index_search_bm25(query, topk=100)
        # print(len(doc_list))
        rank = 1001  # Initialize rank to a value higher than our search range
        
        for j in range(len(doc_list)):
            if has_answers(doc_list[j], ground_truth, _tokenizer, regex=False):
                rank = j + 1
                break

        # print(rank)
        
        recall_at_1.append(1) if rank <= 1 else recall_at_1.append(0)
        recall_at_5.append(1) if rank <= 5 else recall_at_5.append(0)
        recall_at_10.append(1) if rank <= 10 else recall_at_10.append(0)
        recall_at_20.append(1) if rank <= 20 else recall_at_20.append(0)
        recall_at_50.append(1) if rank <= 50 else recall_at_50.append(0)
        recall_at_100.append(1) if rank <= 100 else recall_at_100.append(0)
            
        try:
            print(f"Recall@1: {sum(recall_at_1) / len(recall_at_1)}")
            print(f"Recall@5: {sum(recall_at_5) / len(recall_at_5)}")
            print(f"Recall@20: {sum(recall_at_20) / len(recall_at_20)}")
            print(f"Recall@100: {sum(recall_at_100) / len(recall_at_100)}")
        except:
            continue
        

if __name__ == "__main__":
    main()
