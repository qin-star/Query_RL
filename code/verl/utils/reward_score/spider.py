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

from src.sql.spider import SpiderDatabaseSearcher



_searcher = None


def get_searcher():
    global _searcher
    _searcher = SpiderDatabaseSearcher()
    return _searcher
    

def extract_database_schema(solution_str):
    """Extract the database schema from the solution string."""
    processed_str = solution_str.split("Database Schema:", 1)[1].strip()
    processed_str = processed_str.split("External Knowledge:", 1)[0].strip()
    
    return processed_str

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

    # processed_str = '<think> </think>' + processed_str
    
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
        required_keys = {"sql"}
        if not all(key in data for key in required_keys):
            if do_print:
                print("[Error] Missing required keys in JSON")
            return False

        return True
    # except json.JSONDecodeError:
    except Exception as e:
        if do_print:
            print("[Error] JSON decoding failed")
        return False

def execute_sql(sql_query, db_path):
    """Retrieve items from the search system."""
    searcher = get_searcher()
    results = searcher.search(sql_query, db_path)
    return results
    
def calculate_answer_score(pred_sql, gold_sql, db_path, do_print=False):
    """Calculate answer score based on final_prediction idx."""
    try:
        pred_results = execute_sql(pred_sql, db_path)
        gold_results = execute_sql(gold_sql, db_path)
        
        # answer_score = 2 if set(pred_results) == set(gold_results) else 0.5
        answer_score = 1 if set(pred_results) == set(gold_results) else 0
        
    except Exception as e:
        if do_print:
            print(f"[Error] Error in executing SQL: {e}")
        pred_results = []
        gold_results = []

        answer_score = 0
        # if 'syntax' in str(e):
        #     answer_score = 0
        # else:
        #     answer_score = 0.1

    if do_print:
        # print(f"Retrieved results: {pred_results}")
        # print(f"Target: {gold_results} ")
        print(f"Answer score: {answer_score}")

    
    return answer_score

def compute_score(solution_str, ground_truth, data_source, db_path, format_reward=0.1, answer_reward=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    gold_sql = ground_truth['target']
    
    answer_text, processed_str = extract_solution(solution_str)
    
    do_print = random.randint(1, 16) == 1

    # Validate response structure
    response_format_correct = validate_response_structure(processed_str, do_print)
    json_format_correct = check_json_format(answer_text, do_print)
    format_correct = response_format_correct and json_format_correct
    
    format_score = format_reward if format_correct else -2
    # if do_print:
    #     print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    #     print(f"Format score: {format_score}")
    
    if do_print:
        print(f"--------------------------------")
        # schema = extract_database_schema(solution_str)
        # print(f"Database Schema: {schema}")
        print(f"Solution string: {solution_str}")
    
    answer_score = 0

    if format_correct and answer_text:
        pred_sql = json.loads(answer_text)['sql']
        answer_score = calculate_answer_score(pred_sql, gold_sql, db_path, do_print)

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