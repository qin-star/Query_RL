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
from src.Lucene.utils import ndcg_for_rank


# Instead of initializing at import time, we'll create a global variable but initialize it later
_searcher = None
_tokenizer = SimpleTokenizer()

def get_searcher():
    """Lazily initialize and return the searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
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


search_json_schema = {
    'title': 'search',
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            # "maxLength": 1000
        }
    },
    "required": ["query"]
}


def extract_json_from_llm_output(text):
    pattern = r"```json\n([\s\S]+?)\n```"
    matched_jsons = re.findall(pattern, text)
    
    if matched_jsons:
        extracted_json = matched_jsons[-1]  # get the final one
        return json.loads(extracted_json)
    else:
        # backup plan
        pattern = r"\{.*?\}"
        matched_jsons = re.findall(pattern, text, re.DOTALL)
        
        if matched_jsons:
            extracted_json = matched_jsons[-1]  # get the final one
            return json.loads(extracted_json)
        else:
            raise ValueError('No JSON structure found.')
        
        
def run_index_search_bm25(search_query, topk=50):
    
    searcher = get_searcher()
    
    # Rate limit checking
    hits = searcher.search(search_query, k=topk)
    
    doc_list = [json.loads(hit.lucene_document.get('raw'))['contents'] for hit in hits]
    
    return doc_list
    
def calculate_answer_score_scale(answer_text, label, do_print=False):
    """Calculate answer score based on answer span rank."""
    try:
        data = json.loads(answer_text)
        pred_query = data["query"]
        
        doc_list = run_index_search_bm25(pred_query, topk=3000)
        
        rank = 3001

        # Only need to check up to rank 100 since that's the last meaningful score threshold
        for i in range(len(doc_list)):
            assert isinstance(label, list)
            if has_answers(doc_list[i], label, _tokenizer, regex=False):
                rank = i + 1
                break
            
        if do_print:
            print(f"Rank: {rank}")
        
        if rank <= 5:
            answer_score = 5
        elif rank <= 20:
            answer_score = 4
        elif rank <= 50:
            answer_score = 2
        elif rank <= 100:
            answer_score = 1
        elif rank <= 1000:
            answer_score = 0.5
        elif rank <= 3000:
            answer_score = 0.1
        else:
            answer_score = -3.5

    except Exception as e:
        print(f"[Error] Error in evaluation: {e}")
        answer_score = -4
    
    return answer_score


def compute_score(solution_str, ground_truth):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    # label is a list of groundtruth pmids
    label = ground_truth['target'].tolist()
    
    answer_text, processed_str = extract_solution(solution_str)
    
    do_print = random.randint(1, 32) == 1

    # Validate response structure
    response_format_correct = validate_response_structure(processed_str, do_print)
    format_correct = response_format_correct

    format_score = 1 if format_correct else -4

    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Target: {str(label)} |")
    
    answer_score = 0
    if format_correct and answer_text:
        answer_score = calculate_answer_score_scale(answer_text, label, do_print)

    total_score = format_score + answer_score

    if do_print:
        print("\n" + "-"*80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("="*80 + "\n")

    return total_score
    