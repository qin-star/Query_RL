import json
from pyserini.search.lucene import LuceneSearcher
import time
import pdb

class PyseriniMultiFieldSearch:
    def __init__(self, index_dir="pyserini_index"):
        """Initialize Pyserini MultiField Searcher"""
        self.searcher = LuceneSearcher(index_dir)
        # self.searcher.set_bm25(1.2, 0.75)  # Set BM25 scoring for ranking

    def batch_search(self, queries, top_k=10, threads=4):
        """
        Perform parallel search across multiple fields using batch_search
        :param queries: List of query strings
        :param top_k: Number of results per query
        :param threads: Number of parallel threads for searching
        :return: Dictionary
        """

        # contents
        field_queries = [
            f"contents:{query}"
            for query in queries
        ]
        
        # Perform batch search in parallel
        results_dict = self.searcher.batch_search(
            field_queries,  # List of queries
            [str(i) for i in range(len(queries))],  # Unique query IDs
            k=top_k,
            threads=threads  # Enable parallel searching
        )
        
        final_results = {}
        for i, query in enumerate(queries):
            hits = results_dict[str(i)]  # Get results for query `i`
            formatted_results = [
                (json.loads(hit.raw)["id"], json.loads(hit.raw)["contents"], hit.score)
                for hit in hits
            ]
            final_results[query] = formatted_results

        return final_results

# Example Usage
if __name__ == "__main__":
    search_system = PyseriniMultiFieldSearch(index_dir='code/data/local_index_search/fever/pyserini_index')
    
    queries = [
        "Organic Skin Care",
    ]
    
    tic = time.time()
    search_results = search_system.batch_search(queries, top_k=10, threads=32)
    print(f"Search time: {time.time() - tic:.2f}s")
    # Print results
    for query, results in search_results.items():
        print(f"\n🔍 Query: {query}")
        for asin, text, score in results:
            print(f"  ID: {asin}, TEXT: {text}, Score: {score}")
