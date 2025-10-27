import re
import random
import numpy as np
import ast
import operator
import pdb
import json
import sys
import os
sys.path.append('./')

import torch
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder
from src.Lucene.utils import ndcg_at_k

# REPLACE THIS WITH YOUR OWN INDEX PATH
# index_dir = "/shared/eng/pj20/lmr_model/raw_data/msmarco/indexes/lucene-index-msmarco-passage"

index_dir = "/home/azureuser/cloudfiles/code/DeepRetrieval/indexes/contriever-msmarco-passage-dense-index"
query_encoder_name = "facebook/contriever"

# index_dir = "/home/azureuser/cloudfiles/code/DeepRetrieval/indexes/minilm-msmarco-passage-dense-index"
# query_encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

# index_dir = "/home/azureuser/cloudfiles/code/DeepRetrieval/indexes/mpnet-msmarco-passage-dense-index"
# query_encoder_name = "sentence-transformers/all-mpnet-base-v2"

_searcher = None


def get_searcher(mode='sparse'):
    global _searcher
    if _searcher is None and mode == 'sparse':
        if not os.path.exists(index_dir):
            # print("[Warning] Pyserini index not found for scifact")
            _searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        else:
            _searcher = LuceneSearcher(index_dir=index_dir)
    if _searcher is None and mode == 'dense':
        if not os.path.exists(index_dir):
            _searcher = FaissSearcher.from_prebuilt_index('msmarco-v1-passage.tct_colbert', None)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if 'contriever' in query_encoder_name:
                query_encoder = AutoQueryEncoder(query_encoder_name, pooling='mean', device=device)
            else:
                query_encoder = AutoQueryEncoder(query_encoder_name, pooling='cls', device=device)
            _searcher = FaissSearcher(index_dir=index_dir, query_encoder=query_encoder)
    return _searcher
    

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
        

def validate_response_structure(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")
    
    return validation_passed

def check_json_format(json_str, do_print=False):
    """Check if the given string is a valid JSON and follows the expected structure."""
    try:
        if not json_str:
            if do_print:
                print("[Error] Empty JSON string")
            return False
        
        data = json.loads(json_str)
        
        # Required keys
        required_keys = {"query"}
        if not all(key in data for key in required_keys):
            if do_print:
                print("[Error] Missing required keys in JSON")
            return False

        return True
    except json.JSONDecodeError:
        if do_print:
            print("[Error] JSON decoding failed")
        return False

def retriver_items(query, top_k=3000, mode='sparse'):
    """Retrieve items from the search system."""
    searcher = get_searcher(mode=mode)
    hits = searcher.search(query, k=top_k)
    if mode == 'sparse':
        doc_list = [json.loads(hit.lucene_document.get('raw'))['id'] for hit in hits]
    elif mode == 'dense':
        doc_list = [hit.docid for hit in hits]
    return doc_list
    
def calculate_answer_score(json_str, label, scores, top_k, test_k, mode='sparse', do_print=False):
    """Calculate answer score based on final_prediction idx."""
    try:
        if top_k != 10:
            data = json.loads(json_str)
            query = data['query']
            targets = [str(l) for l in label]
            results = retriver_items(query, top_k=top_k, mode=mode)
            
            
            hit_at_3000 = len(set(results[:3000]) & set(targets))
            recall_at_3000 = hit_at_3000 / len(targets)
            
            hit_at_100 = len(set(results[:100]) & set(targets))
            recall_at_100 = hit_at_100 / len(targets)
            
            hit_at_10 = len(set(results[:10]) & set(targets))
            recall_at_10 = hit_at_10 / len(targets)
            
            answer_score = 0
            
            if recall_at_10 >= 0.5:
                answer_score = 5
            elif recall_at_10 >= 0.4:
                answer_score = 4
            elif recall_at_10 >= 0.3:
                answer_score = 3
            elif recall_at_10 >= 0.2:
                answer_score = 2
            elif recall_at_100 >= 0.5:
                answer_score += 1
            elif recall_at_100 >= 0.4:
                answer_score += 0.5
            elif recall_at_100 >= 0.3:
                answer_score += 0.3
            elif recall_at_3000 >= 0.5:
                answer_score += 0.1
            else:
                answer_score = -3.5

            
            if do_print:
                print(f"Retrieved results: {results}")
                print(f"Target: {label} ")
                print(f"Recall@10: {recall_at_10}")
                print(f"Recall@100: {recall_at_100}")
                print(f"Recall@3000: {recall_at_3000}")
                
        else:
            data = json.loads(json_str)
            query = data['query']
            targets = [str(l) for l in label]
            results = retriver_items(query, top_k=top_k, mode=mode)
            ndcg_score = ndcg_at_k(results, targets, top_k, rel_scores=scores)
            answer_score = ndcg_score
                
                
    except Exception as e:
        if do_print:
            print(f"[Error] Error in evaluation: {e}")
        answer_score = -4
    
    return answer_score

def compute_score(solution_str, ground_truth, data_source, format_reward=0.1, answer_reward=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    label = ground_truth['target']
    scores = [1 for _ in range(len(label))]
    
    answer_text, processed_str = extract_solution(solution_str)
    
    do_print = random.randint(1, 16) == 1

    # Validate response structure
    response_format_correct = validate_response_structure(processed_str, do_print)
    json_format_correct = check_json_format(answer_text, do_print)
    format_correct = response_format_correct and json_format_correct
    
    format_score = 1 if format_correct else -4
    # if do_print:
    #     print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    #     print(f"Format score: {format_score}")
    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
    
    answer_score = 0
    if 'test' in data_source or 'val' in data_source:
        top_k = 10
    else:
        top_k = 3000
        
    if 'sparse' in data_source:
        mode = 'sparse'
    elif 'dense' in data_source:
        mode = 'dense'
    
    test_k = 10
    if format_correct and answer_text:
        answer_score = calculate_answer_score(answer_text, label, scores, top_k, test_k, mode, do_print)

    if answer_score > 0:
        total_score = format_score + answer_score
    else:
        if format_score > 0:
            total_score = 0
        else:
            total_score = format_score
    
    if do_print:
        print("\n" + "-"*80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("="*80 + "\n")

    return total_score


if __name__ == '__main__':
    solution_str = """<|im_start|>assistant:  <answer>{"query": "Microstructural development of human"}</answer>
"""
    ground_truth = {'target': '4983'}
    scores = compute_score(solution_str, ground_truth)
    print(scores)