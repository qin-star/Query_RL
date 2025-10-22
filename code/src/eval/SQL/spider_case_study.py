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

from src.sql.spider import SpiderDatabaseSearcher



_searcher = None


def get_searcher():
    global _searcher
    _searcher = SpiderDatabaseSearcher()
    return _searcher
    

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    model.eval()
    return tokenizer, model

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

def evaluate_model(model, tokenizer, data_path, device, model_name, save_dir, batch_size=8, with_reasoning=True, data_id=None):
    df = pd.read_parquet(data_path)
    
    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['reward_model'].apply(lambda x: x['ground_truth']['target']).tolist()
    db_paths = df['extra_info'].apply(lambda x: x['db_path']).tolist()
    
    model = model.to(device)
    error_count = 0
    execution_scores = []
    sampled_text = []
    
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]


        if batch_start != data_id:
            continue
        
        for i in range(len(batch_inputs)):
            if not with_reasoning:
                batch_inputs[i] = batch_inputs[i].replace("You first think about the reasoning process in the mind and then provides the user with the answer.", "You need to provide the user with the answer.")
                batch_inputs[i] = batch_inputs[i].replace("Show your work in <think> </think> tags. ", "")
                batch_inputs[i] = batch_inputs[i].replace("<think>\n[thinking process]\n</think>", "")
                batch_inputs[i] = batch_inputs[i].replace("<think>", "")

        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)


        with torch.no_grad():
            output_ids = model.generate(**tokenized_inputs, max_new_tokens=512)
        
        for i, output in enumerate(output_ids):
            try:
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                sampled_text.append(generated_text)

                idx = batch_start + i
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

                        # case study
                        if score == 0.0:
                            print('Error!')
                        else:
                            print("--------------------------------")
                            print('True!')
                            print(f'data id: {idx}')
                            print('data_input:')
                            print(batch_inputs[i])
                            print("--------------------------------")
                            print(f'data id: {idx}')
                            print('generated_text:')
                            print(generated_text)
                            print("--------------------------------")
                        
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
        "model_name": model_name,
        "execution_accuracy": final_accuracy,
        "error_count": error_count,
        "total_samples": len(inputs)
    }
    
    with open(os.path.join(save_dir, f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(save_dir, f"{model_name}_sampled_text.json"), "w") as f:
        json.dump(sampled_text, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/dev/ana/training_outputs/spider_7b_coder/actor/global_step_400")
    parser.add_argument("--data_path", type=str, default="data/sql/spider/test.parquet")
    parser.add_argument("--model_name", type=str, default="spider-case-study")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--with_reasoning", type=str, default='True')
    parser.add_argument("--data_id", type=int, default=None)
    args = parser.parse_args()
    
    args.with_reasoning = True if args.with_reasoning.lower() == "true" else False
    
    print(f'args.with_reasoning: {args.with_reasoning}')

    data_id = args.data_id


    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path)
    evaluate_model(model, tokenizer, args.data_path, device, args.model_name, args.save_dir, args.batch_size, args.with_reasoning, data_id)

if __name__ == "__main__":
    main()