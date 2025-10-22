import sys
import os
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import re
# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.sql.bird import BirdDatabaseSearcher
from src.utils.gpt_azure import gpt_chat_4o, gpt_chat_35_msg
from src.utils.gpt import gpt_chat
from src.utils.claude_api import get_claude_response




def get_llm_response(prompt, llm_name):
    if llm_name == "gpt-4o":
        # return gpt_chat_4o(prompt)
        return gpt_chat('gpt-4o', prompt)
    elif llm_name == "gpt-35":
        # return gpt_chat_35_msg(prompt)
        return gpt_chat('gpt-3.5-turbo', prompt)
    elif llm_name == "claude-3-haiku":
        return get_claude_response("haiku", prompt)
    elif llm_name == "claude-35-sonnet":
        return get_claude_response("sonnet", prompt)
    else:
        raise ValueError(f"Unknown LLM: {llm_name}")



_searcher = None


def get_searcher():
    global _searcher
    _searcher = BirdDatabaseSearcher()
    return _searcher
    

def extract_solution(solution_str):
    """Extract the SQL query from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     processed_str = solution_str.split("Assistant:", 1)[1].strip()
    # elif "<|im_start|>assistant" in solution_str:
    #     processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    # else:
    #     print("[Error] Failed to locate model response header")
    #     return None, processed_str
    processed_str = solution_str

    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)
    
    if matches:
        return matches[-1].strip(), processed_str
    else:
        print("[Error] No valid answer tags found")
        return None, processed_str

def execute_sql(sql_query, db_path):
    """Execute SQL query and return results."""
    searcher = get_searcher()
    results = searcher.search(sql_query, db_path)
    return results

def calculate_execution_score(pred_sql, gold_sql, db_path):
    """Calculate score based on SQL execution results."""
    try:
        pred_results = execute_sql(pred_sql, db_path)
        gold_results = execute_sql(gold_sql, db_path)
        
        # Compare results as sets to ignore order
        execution_score = 1.0 if set(pred_results) == set(gold_results) else 0.0
        
    except Exception as e:
        print(f"[Error] Error in executing SQL: {e}")
        execution_score = 0.0

    return execution_score

def evaluate_model(llm_name, data_path, save_dir, with_reasoning=True):
    df = pd.read_parquet(data_path)
    
    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['reward_model'].apply(lambda x: x['ground_truth']['target']).tolist()
    db_paths = df['extra_info'].apply(lambda x: x['db_path']).tolist()
    
    error_count = 0
    execution_scores = []
    sampled_text = []
    
    for idx, data_input in tqdm(enumerate(inputs), desc="Evaluating", total=len(inputs)):
        try:
            if not with_reasoning:
                data_input = data_input.replace("You first think about the reasoning process in the mind and then provides the user with the answer.", "You need to provide the user with the answer.")
                data_input = data_input.replace("Show your work in <think> </think> tags. ", "")
                data_input = data_input.replace("<think>\n[thinking process]\n</think>", "")
                data_input = data_input.replace("<think>", "")
                data_input = data_input.split("<|im_start|>user")[1]

            data_input = data_input + '\nYour output of SQL query must be in one line.'
            
            generated_text = get_llm_response(data_input, llm_name)
            sampled_text.append(generated_text)

            answer_text, processed_str = extract_solution(generated_text)
            if answer_text:
                try:
                    pred_sql = json.loads(answer_text)['sql']
                    score = calculate_execution_score(
                        pred_sql,
                        targets[idx],
                        db_paths[idx]
                    )
                    execution_scores.append(score)

                    # # case study
                    # if score == 0.0:
                    #     print("--------------------------------")
                    #     print(f'data id: {idx}')
                    #     print('data_input:')
                    #     print(data_input)
                    #     print("--------------------------------")
                    #     print(f'data id: {idx}')
                    #     print('generated_text:')
                    #     print(generated_text)
                    #     print("--------------------------------")
                    #     input('Next?')

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[Error] JSON parsing error: {e}")
                    execution_scores.append(0.0)
                    error_count += 1
            else:
                execution_scores.append(0.0)
                error_count += 1
            
        except Exception as e:
            print(f"[Error] Evaluation error: {e}")
            execution_scores.append(0.0)
            error_count += 1
            continue

        # Print intermediate results
        if len(execution_scores) > 0:
            print(f"Current Execution Accuracy: {sum(execution_scores) / len(execution_scores):.4f}")
    
    # Calculate and print final metrics
    final_accuracy = sum(execution_scores) / len(execution_scores)
    print(f"\nFinal Results:")
    print(f"Execution Accuracy: {final_accuracy:.4f}")
    print(f"Error Count: {error_count}")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    results = {
        "model_name": llm_name,
        "execution_accuracy": final_accuracy,
        "error_count": error_count,
        "total_samples": len(inputs)
    }
    
    with open(os.path.join(save_dir, f"{llm_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(save_dir, f"{llm_name}_sampled_text.json"), "w") as f:
        json.dump(sampled_text, f, indent=2)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument("--data_path", type=str, default="data/sql/bird/test.parquet")
    parser.add_argument("--save_dir", type=str, default="results/sql/bird")
    parser.add_argument("--with_reasoning", type=str, default='True')
    # parser.add_argument("--with_reasoning", type=str, default='false')
    args = parser.parse_args()
    
    args.with_reasoning = True if args.with_reasoning.lower() == "true" else False

    # llm_name = "gpt-35"
    llm_name = "gpt-4o"
    # llm_name = "claude-3-haiku"
    # llm_name = "claude-35-sonnet"

    args.llm_name = llm_name

    evaluate_model(args.llm_name, args.data_path, args.save_dir, args.with_reasoning)


if __name__ == "__main__":
    main()
