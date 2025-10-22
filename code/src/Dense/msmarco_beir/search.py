from pyserini.search.faiss import FaissSearcher
import time
import pdb

class PyseriniFaissSearcher:
    def __init__(self, index_dir, model_name):
        """
        Initialize the Pyserini FAISS searcher.
        :param index_dir: Path to the FAISS index directory.
        :param model_name: Name of the transformer model for query encoding.
        """
        self.index_dir = index_dir
        self.model_name = model_name
        self.searcher = FaissSearcher(index_dir, query_encoder=model_name)
        # self.searcher.query_encoder.model.to('cuda')
        # self.searcher.query_encoder.device = 'cuda'
    
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

# Example usage
if __name__ == "__main__":
    index_dir = 'code/data/local_index_search/nfcorpus/dense_index/faiss-flat.beir-v1.0.0-nfcorpus.bge-base-en-v1.5.20240107'
    model_name = 'BAAI/bge-base-en-v1.5'
    searcher = PyseriniFaissSearcher(index_dir, model_name)
    
    # Single query example
    query = "How does white matter develop in the human brain?"
    time_start = time.time()
    top_results = searcher.search(query, top_k=5)
    time_end = time.time()
    print(f"\n⏱️ Search time: {time_end - time_start:.4f} seconds")
    
    pdb.set_trace()
    print("\n🔍 Top Search Results:")
    for doc_id, score in top_results:
        print(f"📄 Doc ID: {doc_id} | 🔢 Score: {score:.4f}")
    
    # Batch search example
    queries = [
        "How does white matter develop in the human brain?",
        "What are the effects of exercise on cognitive function?",
        "Can diet influence neurodegenerative diseases?"
    ]
    
    time_start = time.time()
    batch_results = searcher.batch_search(queries, top_k=5)
    time_end = time.time()
    print(f"\n⏱️ Batch search time: {time_end - time_start:.4f} seconds")
    
    pdb.set_trace()
    print("\n🔍 Top Batch Search Results:")
    for query, results in batch_results.items():
        print(f"\n🔎 Query: {query}")
        for doc_id, score in results:
            print(f"📄 Doc ID: {doc_id} | 🔢 Score: {score:.4f}")
