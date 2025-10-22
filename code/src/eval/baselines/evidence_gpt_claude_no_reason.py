import sys
import os
import concurrent.futures
import time
from functools import partial

# Debug: Print current directory and Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Remove one dirname call to point to Panacea-R1
# Add the project root to Python path
sys.path.insert(0, project_root)  # This will now add Panacea-R1 to the path

import argparse
import pandas as pd
from tqdm import tqdm
import json
import re

import utils.java_init

# Add these at the top with other global variables
from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from tqdm import tqdm
from src.utils.gpt_azure import gpt_chat_4o, gpt_chat_35_msg
from src.utils.claude_api import get_claude_response


_searcher = None
_tokenizer = SimpleTokenizer()

def get_searcher():
    """Lazily initialize and return the searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
    return _searcher


def extract_solution(response):
    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, response, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip() # Return the last matched answer
    else:
        print(f"Warning: No answer tag found in the response. Using the response itself as the answer.")
        return response
    
    
def extract_json_from_llm_output(text):
    pattern = r"```json\n([\s\S]+?)\n```"
    matched_jsons = re.findall(pattern, text)
    
    if matched_jsons:
        extracted_json = matched_jsons[-1]  # get the final one
        return json.loads(extracted_json)['query']
    else:
        try:
            pattern = r"\{.*?\}"
            matched_jsons = re.findall(pattern, text, re.DOTALL)
            
            if matched_jsons:
                extracted_json = matched_jsons[-1]  # get the final one
                try:
                    print(f"Extracted JSON: {extracted_json}")
                    return json.loads(extracted_json)['query']
                except:
                    print(f"Warning: No JSON structure found. Using the response itself as the query. {extracted_json}")
                    return extracted_json.split(": ")[1].strip()
            else:
                raise ValueError('No JSON structure found.')
        except:
            return text
        
def run_index_search_bm25(search_query, topk=50):
    
    searcher = get_searcher()
    
    # Rate limit checking
    hits = searcher.search(search_query, k=topk)
    
    doc_list = [json.loads(hit.lucene_document.get('raw'))['contents'] for hit in hits]
    
    return doc_list

# Do not inject your own knowledge into the query. You should assume you don't know the answer and just use the query itself to find the answer.
def process_prompt(prompt):
    prompt = prompt.replace("""<|im_start|>system\nYou are a helpful assistant. You directly provide the user with the answer.<|im_end|>\n<|im_start|>user\n""", "")
    prompt = prompt.replace("Note: ", "Note: The content between <answer> and </answer> should be a valid JSON object, e.g., using double quotes for strings, using slashes for special characters.")
    return prompt

def compute_filtered_recall(recall_list, error_count, total_queries):
    """
    Compute filtered recall by excluding error cases.
    
    Args:
        recall_list: List of recall values (0s and 1s)
        error_count: Number of errors encountered
        total_queries: Total number of queries processed
    
    Returns:
        tuple: (filtered_recall, raw_recall)
    """
    successful_queries = total_queries - error_count
    if successful_queries == 0:
        return 0.0, 0.0
    
    raw_recall = sum(recall_list) / total_queries
    filtered_recall = sum(recall_list) / successful_queries
    return filtered_recall, raw_recall

def save_generations(responses, model_name, dataset_name):
    """Save generation results to a JSON file."""
    output_dir = "results/generations_no_reasoning"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_{model_name}_generations.json")
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)
    print(f"Saved generations to {output_file}")

def save_metrics(metrics, model_name, dataset_name):
    """Save evaluation metrics to a text file."""
    output_dir = "results/metrics_no_reasoning"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_{model_name}_metrics.txt")
    with open(output_file, 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value}\n")
    print(f"Saved metrics to {output_file}")

def evaluate_model(data_path, model_name, dataset_name, batch_size=8, max_retries=5):
    print(f"\nEvaluating {model_name} on {dataset_name}")
    df = pd.read_parquet(data_path)
    
    all_responses = []
    
    inputs = [process_prompt(item[0]['content']) for item in df['prompt'].tolist()]
    targets = df['label'].tolist()
    
    error_count = 0
    total_queries = len(inputs)
    
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    recall_at_20 = []
    recall_at_50 = []
    recall_at_100 = []
    
    def process_single_prompt_with_retry(prompt, retry_count=0):
        try:
            if 'gpt4o' in model_name:
                return gpt_chat_4o(prompt=prompt)
            elif 'gpt35' in model_name:
                return gpt_chat_35_msg(prompt=prompt)
            elif 'claude35' in model_name:
                return get_claude_response(llm="sonnet", prompt=prompt)
            elif 'claude3' in model_name:
                return get_claude_response(llm="hiku", prompt=prompt)
        except Exception as e:
            if retry_count < max_retries:
                # Exponential backoff: wait 2^retry_count seconds before retrying
                wait_time = 2 ** retry_count
                print(f"API call failed. Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return process_single_prompt_with_retry(prompt, retry_count + 1)
            else:
                print(f"Failed after {max_retries} attempts. Error: {str(e)}")
                return None

    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        
        # Use ThreadPoolExecutor for parallel API calls with retries
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Create futures with their original indices
            futures = []
            for idx, prompt in enumerate(batch_inputs):
                future = executor.submit(process_single_prompt_with_retry, prompt)
                futures.append((idx, future))
            
            # Initialize response list with None values
            responses = [None] * len(batch_inputs)
            
            # Process futures as they complete while maintaining original order
            for idx, future in futures:
                try:
                    result = future.result()
                    if result is not None:
                        responses[idx] = result
                    else:
                        responses[idx] = None
                        error_count += 1
                except Exception as e:
                    print(f"Unexpected error processing prompt at index {idx}: {str(e)}")
                    responses[idx] = None
                    error_count += 1
            
        all_responses.extend(responses)
        
        for i, response in enumerate(responses):
            try:
                if response is None:
                    # Skip processing for failed responses
                    recall_at_1.append(0)
                    recall_at_5.append(0)
                    recall_at_10.append(0)
                    recall_at_20.append(0)
                    recall_at_50.append(0)
                    recall_at_100.append(0)
                    continue
                    
                generated_text = response
                idx = batch_start + i
                # convert target from ndarray to list
                
                extracted_solution = extract_solution(generated_text)
                query = extract_json_from_llm_output(extracted_solution)
                
                rank = 1001
                target = targets[idx].tolist()
                searched_doc_list = run_index_search_bm25(query, topk=100)
                
                for j in range(len(searched_doc_list)):
                    if has_answers(searched_doc_list[j], target, _tokenizer, regex=False):
                        rank = j + 1
                        break
                    
                recall_at_1.append(1) if rank <= 1 else recall_at_1.append(0)
                recall_at_5.append(1) if rank <= 5 else recall_at_5.append(0)
                recall_at_10.append(1) if rank <= 10 else recall_at_10.append(0)
                recall_at_20.append(1) if rank <= 20 else recall_at_20.append(0)
                recall_at_50.append(1) if rank <= 50 else recall_at_50.append(0)
                recall_at_100.append(1) if rank <= 100 else recall_at_100.append(0)
                
            except:
                recall_at_1.append(0)
                recall_at_5.append(0)
                recall_at_10.append(0)
                recall_at_20.append(0)
                recall_at_50.append(0)
                recall_at_100.append(0)
                error_count += 1
                print(f"Error: {generated_text}, Error count: {error_count}")
                continue
            
            
        try:
            print(f"Current Recall@1: {sum(recall_at_1) / len(recall_at_1)}")
            print(f"Current Recall@5: {sum(recall_at_5) / len(recall_at_5)}")
            print(f"Current Recall@20: {sum(recall_at_20) / len(recall_at_20)}")
            print(f"Current Recall@100: {sum(recall_at_100) / len(recall_at_100)}")
        except:
            continue

    # Save generations
    save_generations(all_responses, model_name, dataset_name)
    
    # Calculate final metrics
    metrics = {
        "recall@1": sum(recall_at_1) / len(recall_at_1),
        "recall@5": sum(recall_at_5) / len(recall_at_5),
        "recall@10": sum(recall_at_10) / len(recall_at_10),
        "recall@20": sum(recall_at_20) / len(recall_at_20),
        "recall@50": sum(recall_at_50) / len(recall_at_50),
        "recall@100": sum(recall_at_100) / len(recall_at_100),
        "error_count": error_count,
        "total_queries": total_queries,
        "success_rate": (total_queries - error_count) / total_queries
    }
    
    # Save metrics
    save_metrics(metrics, model_name, dataset_name)
    
    return metrics

def check_if_processed(model_name, dataset_name):
    """Check if this combination has already been processed by looking for both metrics and generations files."""
    metrics_file = os.path.join("results/metrics_no_reasoning", f"{dataset_name}_{model_name}_metrics.txt")
    generations_file = os.path.join("results/generations_no_reasoning", f"{dataset_name}_{model_name}_generations.json")
    
    if os.path.exists(metrics_file) and os.path.exists(generations_file):
        try:
            # Verify the files are valid by trying to read them
            with open(metrics_file, 'r') as f:
                metrics_content = f.read()
            with open(generations_file, 'r') as f:
                json.load(f)
            # If both files exist and are valid, consider it processed
            return True
        except:
            # If there's any error reading the files, consider it not processed
            return False
    return False

def load_existing_results():
    """Load existing overall results if they exist."""
    summary_file = "results/overall_summary_evi_no_reasoning.json"
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_concurrent_models", type=int, default=2)
    args = parser.parse_args()

    # Define all combinations
    datasets = {
        "nq_serini": "data/local_index_search/no_reason/nq_serini/test.parquet",
        "triviaqa": "data/local_index_search/no_reason/triviaqa/test.parquet",
        "squad": "data/local_index_search/no_reason/squad/test.parquet"
    }
    
    # Group models by API type to avoid rate limiting conflicts
    gpt_models = ["gpt35", "gpt4o"]
    claude_models = ["claude3", "claude35"]
    model_groups = [claude_models, gpt_models]
    
    # Create results directories
    os.makedirs("results/generations_no_reasoning", exist_ok=True)
    os.makedirs("results/metrics_no_reasoning", exist_ok=True)
    
    # Load existing results
    all_results = load_existing_results()
    
    def evaluate_model_wrapper(dataset_name, data_path, model_name, batch_size):
        # Skip if already processed
        if check_if_processed(model_name, dataset_name):
            print(f"\nSkipping {model_name} on {dataset_name} - already processed")
            try:
                # Try to load existing metrics
                with open(os.path.join("results/metrics_no_reasoning", f"{dataset_name}_{model_name}_metrics.txt"), 'r') as f:
                    metrics_content = f.read()
                print(f"Existing metrics:\n{metrics_content}")
                return dataset_name, model_name, None
            except:
                return dataset_name, model_name, None
        
        try:
            metrics = evaluate_model(data_path, model_name, dataset_name, batch_size)
            print(f"\nResults for {model_name} on {dataset_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value}")
            return dataset_name, model_name, metrics
        except Exception as e:
            print(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
            return dataset_name, model_name, None

    for dataset_name, data_path in datasets.items():
        if dataset_name not in all_results:
            all_results[dataset_name] = {}
        
        # Process each group of models sequentially to avoid API conflicts
        for model_group in model_groups:
            # Create evaluation tasks for this group
            eval_tasks = [(dataset_name, data_path, model_name) 
                         for model_name in model_group 
                         if not check_if_processed(model_name, dataset_name)]
            
            if not eval_tasks:  # Skip if all models in this group are already processed
                continue
                
            # Run models in this group in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_group)) as executor:
                futures = [executor.submit(evaluate_model_wrapper, task[0], task[1], task[2], args.batch_size) 
                          for task in eval_tasks]
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    dataset, model, metrics = future.result()
                    if metrics is not None:
                        all_results[dataset][model] = metrics
    
    # Save overall results summary
    with open("results/overall_summary_evi_no_reasoning.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
