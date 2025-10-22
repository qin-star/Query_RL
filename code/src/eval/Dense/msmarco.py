import sys
import os

# Debug: Print current directory and Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Remove one dirname call to point to Panacea-R1
# Add the project root to Python path
sys.path.insert(0, project_root)  # This will now add Panacea-R1 to the path


import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import re

# Add these at the top with other global variables
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder
from tqdm import tqdm
from src.Lucene.utils import ndcg_at_k


# REPLACE THIS WITH YOUR OWN INDEX PATH
index_dir = "/home/azureuser/cloudfiles/code/DeepRetrieval/indexes/contriever-msmarco-passage-dense-index"
query_encoder_name = "facebook/contriever"

_searcher = None

def get_searcher(mode='dense'):
    global _searcher
    if _searcher is None and mode == 'dense':
        if not os.path.exists(index_dir):
            print("[Warning] Pyserini index not found")
            _searcher = FaissSearcher.from_prebuilt_index('msmarco-v1-passage', None)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if 'contriever' in query_encoder_name:
                query_encoder = AutoQueryEncoder(query_encoder_name, pooling='mean', device=device)
            else:
                query_encoder = AutoQueryEncoder(query_encoder_name, pooling='cls', device=device)
            _searcher = FaissSearcher(index_dir=index_dir, query_encoder=query_encoder)
    return _searcher


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    model.eval()
    return tokenizer, model


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    else:
        print("[Error] Failed to locate model response header")
        return None, processed_str

    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip(), processed_str  # Return the last matched answer
    else:
        print("[Error] No valid answer tags found")
        return None, processed_str
    
        
def retriver_items(query, top_k=3000, mode='dense'):
    """Retrieve items from the search system."""
    searcher = get_searcher(mode=mode)
    hits = searcher.search(query, k=top_k)
    if mode == 'sparse':
        doc_list = [json.loads(hit.lucene_document.get('raw'))['id'] for hit in hits]
    elif mode == 'dense':
        doc_list = [hit.docid for hit in hits]
    return doc_list


def evaluate_model(model, tokenizer, data_path, device, model_name, save_dir, batch_size=1, search_api=None):
    df = pd.read_parquet(data_path)
    
    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['label'].tolist()
    
    model = model.to(device)
    error_count = 0
    
    ndcg_scores = []
    
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        
        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(**tokenized_inputs, max_new_tokens=350)
            # output_ids = model.generate_sequences(**tokenized_inputs, max_new_tokens=350, temperature=0.6, do_sample=False)
        
        for i, output in enumerate(output_ids):
            try:
                generated_text = tokenizer.decode(output)
                idx = batch_start + i
                # convert target from ndarray to list
                
                extracted_solution, processed_str = extract_solution(generated_text)
                query = json.loads(extracted_solution)['query']
                # query = query.lower()
                print(query)
                
                target = targets[idx].tolist()
                scores = [1 for _ in range(len(target))]
                searched_doc_list = retriver_items(query, top_k=100, mode='dense')
                
                ndcg_score = ndcg_at_k(searched_doc_list, target, 10, rel_scores=scores)
                ndcg_scores.append(ndcg_score)
            except Exception as e:
                error_count += 1
                print(f"Error: {e}")
                continue
            

        print(f"NDCG@10: {sum(ndcg_scores) / len(ndcg_scores)}")
        print("Error count: ", error_count)
        
        
        
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/azureuser/cloudfiles/code/DeepRetrieval/training_outputs/msmarco_search_3b_all_dense_contriever/actor/global_step_100")
    parser.add_argument("--data_path", type=str, default="data/local_index_search/msmarco_health/dense/test.parquet")
    parser.add_argument("--model_name", type=str, default="msmarco_health_dense_recall_step_100")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path)
    evaluate_model(model, tokenizer, args.data_path, device, args.model_name, args.save_dir, args.batch_size)

if __name__ == "__main__":
    main()
