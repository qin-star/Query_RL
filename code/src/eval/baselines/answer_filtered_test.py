from tqdm import tqdm
from src.utils.claude_api import get_claude_response

import utils.java_init
import re
# Add these at the top with other global variables
from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
import json
import os
import argparse
import pandas as pd
import time
import concurrent.futures

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
    

def process_llm_query_with_retry(prompt, retry_count=0, max_retries=3):
    """Process a single LLM query with retry logic."""
    try:
        if retry_count > 0:
            # Add exponential backoff delay
            wait_time = min(2 ** retry_count, 32)  # Cap at 32 seconds
            time.sleep(wait_time)
        
        response = get_claude_response("sonnet", prompt)
        return response
    except Exception as e:
        if retry_count < max_retries:
            print(f"API call failed. Error: {str(e)}. Retrying ({retry_count + 1}/{max_retries})...")
            return process_llm_query_with_retry(prompt, retry_count + 1, max_retries)
        else:
            print(f"Failed after {max_retries} attempts. Error: {str(e)}")
            return None

def get_if_answer_span_in_query_batch(original_queries, queries, answer_candidates_list, batch_size=10):
    """
    Process multiple queries in parallel using a thread pool.
    
    Args:
        queries: List of query strings
        answer_candidates_list: List of answer candidate lists
        batch_size: Number of concurrent API calls to make
        
    Returns:
        List of LLM responses in the same order as the input queries
    """

    def create_prompt(original_query, query, answer_candidates):
        if type(query) == list:
            query = query[0]
        # print(answer_candidates)
        return """You are a helpful assistant that checks the quality of query augmentation. As we use LLM to augment the query, we need to check if the augmented query can be derived from the original query.
Your task is to analyze if there are answer spans in the query that directly match the answer candidates and cannot be derived from the original query without using prior knowledge.

Instructions:
1. Check if any part of the augmented query exactly matches or paraphrases any answer candidate, if so, set "has_answer" to true.
2. If the augmented query cannot be derived from the original query without using prior knowledge, set "cannot_be_derived" to true.
3. If both "has_answer" and "cannot_be_derived" are true, remove those answer spans from the query, without changing other parts of the augmented query.
4. Return your analysis in a strict JSON format

IMPORTANT: 
You must respond with valid JSON wrapped in <answer> tags. The JSON must have this exact structure:
The content between <answer> and </answer> MUST be a valid JSON object:
- Use double quotes for strings
- Escape special characters with backslashes (e.g., \\" for quotes within strings)
- Follow standard JSON formatting rules

Example valid response:
For the following original query:
"How many people live in New York City in 2020?"

and the augmented query:
("population" OR "residents") AND "new york city" AND "2020" AND "8 million"

Answer candidates to check against:
["8.253 million", "8,253,213", "8.25M", "approximately 8.25 million"]

<answer>
{
    "has_answer": true,
    "cannot_be_derived": true,
    "answer_span_in_query": ["8 million"],
    "matched_answer_candidates": ["approximately 8.25 million", "8.25M"],
    "cleaned_query": "(\"population\" OR \"residents\") AND \"new york city\" AND \"2020\""
}
</answer>

Now, your turn:

Original query:
""" + original_query + """

Augmented query to analyze:
""" + query + """   

Answer candidates to check against:
""" + str(answer_candidates) + """

Your response:
"""

    # Create all prompts upfront
    all_prompts = [create_prompt(oq, q, c) for oq, q, c in zip(original_queries, queries, answer_candidates_list)]
    total_queries = len(all_prompts)
    results = [None] * total_queries  # Pre-allocate results list to maintain order
    
    # Process queries in batches
    with tqdm(total=total_queries, desc="Processing LLM queries") as pbar:
        for batch_start in range(0, total_queries, batch_size):
            batch_end = min(batch_start + batch_size, total_queries)
            batch_indices = list(range(batch_start, batch_end))
            batch_prompts = [all_prompts[i] for i in batch_indices]
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                # Map batch indices to futures
                future_to_idx = {
                    executor.submit(process_llm_query_with_retry, prompt): idx 
                    for idx, prompt in zip(batch_indices, batch_prompts)
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        response = future.result()
                        if response:
                            results[idx] = extract_solution(response)
                        else:
                            print(f"Warning: Null response for query at index {idx}")
                            results[idx] = None
                    except Exception as e:
                        print(f"Error processing query at index {idx}: {str(e)}")
                        results[idx] = None
                    pbar.update(1)
    
    return results

def get_if_answer_span_in_query(query, answer_candidates):
    """Single query wrapper around the batch processing function."""
    results = get_if_answer_span_in_query_batch([query], [answer_candidates], batch_size=1)
    return results[0] if results else None

def extract_json_from_llm_output(text):
    def debug_json_parse(json_str):
        try:
            # First try direct parsing
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # print(f"Debug - JSON parse error: {str(e)}")
            # print(f"Debug - Problem string: {repr(json_str)}")
            
            # Try with proper quote escaping for nested quotes in cleaned_query
            try:
                # First normalize any already escaped quotes
                normalized = json_str.replace('\\"', '"')
                # Then find the cleaned_query value and properly escape its quotes
                pattern = r'"cleaned_query":\s*"(.*?)"(?=\s*[,}])'
                def escape_quotes(match):
                    query = match.group(1)
                    # Escape any quotes that aren't already escaped
                    escaped = query.replace('"', '\\"')
                    return f'"cleaned_query": "{escaped}"'
                
                fixed_json = re.sub(pattern, escape_quotes, normalized, flags=re.DOTALL)
                # print("Fixed JSON (1): ", fixed_json)
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                # If that fails, try the original quote normalization approach
                try:
                    normalized = json_str.replace('\\"', "'")
                    # print("Fixed JSON (2): ", normalized)
                    return json.loads(normalized)
                except json.JSONDecodeError:
                    print("Failed to parse JSON (3): ", json_str)
                    return None

    # First try to find JSON within <answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, text, re.DOTALL)
    
    if answer_matches:
        json_str = answer_matches[-1].strip()
        result = debug_json_parse(json_str)
        if result:
            return result
    
    # Try to find JSON within code blocks
    code_pattern = r'```(?:json)?\s*(.*?)\s*```'
    code_matches = re.findall(code_pattern, text, re.DOTALL)
    
    if code_matches:
        json_str = code_matches[-1].strip()
        result = debug_json_parse(json_str)
        if result:
            return result
    
    # Try to find any JSON-like structure
    json_pattern = r'\{[^{}]*\}'
    json_matches = re.findall(json_pattern, text, re.DOTALL)
    
    if json_matches:
        json_str = json_matches[-1].strip()
        result = debug_json_parse(json_str)
        if result:
            return result
        else:
            try:
                print(f"Warning: Failed to parse JSON from response: {text}")
                result = {
                    "has_answer": json_str.split("\"has_answer\":")[1].split("\"cannot_be_derived\":")[0].strip(),
                    "cannot_be_derived": json_str.split("\"cannot_be_derived\":")[1].split("\"answer_span_in_query\":")[0].strip(),
                    "answer_span_in_query": json_str.split("\"answer_span_in_query\":")[1].split("\"matched_answer_candidates\":")[0].strip(),
                    "matched_answer_candidates": json_str.split("\"matched_answer_candidates\":")[1].split("\"cleaned_query\":")[0].strip(),
                    "cleaned_query": json_str.split("\"cleaned_query\":")[1].split("}", 1)[0].strip()
                }
                print("Result: ", result)
                print("type of result: ", type(result))
                return result
            except Exception as e:
                print(f"Error parsing JSON from response: {str(e)}")
                return None

    print(f"Warning: No JSON structure found in response: {text}")
    return None

def run_index_search_bm25(search_query, topk=50):
    searcher = get_searcher()
    hits = searcher.search(search_query, k=topk)
    doc_list = [json.loads(hit.lucene_document.get('raw'))['contents'] for hit in hits]
    return doc_list

def evaluate_query(query, target, topk=100):
    rank = 1001
    searched_doc_list = run_index_search_bm25(query, topk=topk)
    
    for j in range(len(searched_doc_list)):
        if has_answers(searched_doc_list[j], target, _tokenizer, regex=False):
            rank = j + 1
            break
            
    recall_dict = {
        "recall@1": 1 if rank <= 1 else 0,
        "recall@5": 1 if rank <= 5 else 0,
        "recall@10": 1 if rank <= 10 else 0,
        "recall@20": 1 if rank <= 20 else 0,
        "recall@50": 1 if rank <= 50 else 0,
        "recall@100": 1 if rank <= 100 else 0
    }
    return recall_dict

def process_generations(dataset_name, model_name, generations_path, data_path):
    """Process generations and evaluate cleaned queries."""
    # Load original data
    df = pd.read_parquet(data_path)
    raw_queries_ = df['input'].tolist()
    # print(raw_queries[:10])
    # print("length of raw queries: ", len(raw_queries))
    targets = df['label'].tolist()
    
    # Load generations
    with open(generations_path, 'r') as f:
        generations = json.load(f)
    
    results = {
        "original_metrics": {
            "recall@1": [], "recall@5": [], "recall@10": [],
            "recall@20": [], "recall@50": [], "recall@100": []
        },
        "cleaned_metrics": {
            "recall@1": [], "recall@5": [], "recall@10": [],
            "recall@20": [], "recall@50": [], "recall@100": []
        },
        "injection_stats": {
            "total_queries": len(generations),
            "queries_with_injection": 0,
            "answer_spans": []
        },
        "sample_details": []  # New field to store per-sample information
    }
    
    # Extract original queries and targets first
    original_queries = []
    target_list = []
    raw_queries = []
    
    for idx, response in enumerate(generations):
        try:
            # Extract original query
            original_query = extract_json_from_llm_output(response)['query']
            target = targets[idx].tolist()
            raw_query = str(raw_queries_[idx])
            
            original_queries.append(original_query)
            target_list.append(target)
            raw_queries.append(raw_query)
            
        except Exception as e:
            print(f"Error extracting query {idx}: {str(e)}")
            print(raw_queries_[idx])
            # Add placeholder values
            original_queries.append("error")
            target_list.append(targets[idx].tolist())
            raw_queries.append(str(raw_queries_[idx]))
            
    # Batch process injection checks
    print(f"Checking for knowledge injection in {len(original_queries)} queries...")
    batch_size = 12
    injection_check_results = get_if_answer_span_in_query_batch(
        raw_queries, original_queries, target_list, batch_size=batch_size
    )
    
    # Process the results
    for idx in tqdm(range(len(generations)), desc=f"Processing {dataset_name}-{model_name}"):
        try:
            original_query = original_queries[idx]
            target = target_list[idx]
            
            # Skip completely errored entries
            if original_query == "error":
                continue
            
            # Initialize sample details
            sample_info = {
                "index": idx,
                "original_query": original_query,
                "answer_candidate": target,
                "cleaned_query": original_query,  # Default to original if no cleaning needed
                "has_injection": False
            }
            
            # Evaluate original query
            original_metrics = evaluate_query(original_query, target)
            for k, v in original_metrics.items():
                results["original_metrics"][k].append(v)
            
            # Get the injection check result from the batch processing
            injection_check = injection_check_results[idx]
            
            if injection_check:
                injection_result = extract_json_from_llm_output(injection_check)
                
                if injection_result and "true" in str(injection_result.get("has_answer")).lower() and "true" in str(injection_result.get("cannot_be_derived")).lower():
                    sample_info["has_injection"] = True
                    results["injection_stats"]["queries_with_injection"] += 1
                    results["injection_stats"]["answer_spans"].extend(injection_result.get("answer_span_in_query", []))
                    
                    # Evaluate cleaned query
                    cleaned_query = injection_result.get("cleaned_query", original_query)
                    sample_info["cleaned_query"] = cleaned_query
                    cleaned_metrics = evaluate_query(cleaned_query, target)
                    for k, v in cleaned_metrics.items():
                        results["cleaned_metrics"][k].append(v)
                elif injection_result == None:
                    sample_info["has_injection"] = True
                    results["injection_stats"]["queries_with_injection"] += 1
                    results["injection_stats"]["answer_spans"].extend(target)
                else:
                    # If no injection, use same metrics as original
                    for k, v in original_metrics.items():
                        results["cleaned_metrics"][k].append(v)
            else:
                # If processing failed, use same metrics as original
                for k, v in original_metrics.items():
                    results["cleaned_metrics"][k].append(v)
            
            # Add sample details to results
            results["sample_details"].append(sample_info)
                
        except Exception as e:
            print(f"Error processing query {idx}: {str(e)}")
            # Add 0s for failed cases
            for metric_dict in [results["original_metrics"], results["cleaned_metrics"]]:
                for k in metric_dict:
                    metric_dict[k].append(0)
            # Add error case to sample details
            results["sample_details"].append({
                "index": idx,
                "original_query": original_queries[idx] if idx < len(original_queries) else "error",
                "answer_candidate": target_list[idx] if idx < len(target_list) else [],
                "cleaned_query": "error",
                "has_injection": False,
                "error": str(e)
            })
    
    # Calculate average metrics
    final_results = {
        "dataset": dataset_name,
        "model": model_name,
        "original_metrics": {
            k: sum(v)/len(v) if v else 0 for k, v in results["original_metrics"].items()
        },
        "cleaned_metrics": {
            k: sum(v)/len(v) if v else 0 for k, v in results["cleaned_metrics"].items()
        },
        "injection_stats": {
            "total_queries": results["injection_stats"]["total_queries"],
            "queries_with_injection": results["injection_stats"]["queries_with_injection"],
            "injection_rate": results["injection_stats"]["queries_with_injection"] / results["injection_stats"]["total_queries"] if results["injection_stats"]["total_queries"] > 0 else 0,
            "unique_answer_spans": len(set(results["injection_stats"]["answer_spans"])),
            "answer_spans": list(set(results["injection_stats"]["answer_spans"]))
        },
        "sample_details": results["sample_details"]  # Include all sample details in final results
    }
    
    # Save results
    output_dir = "results/answer_filtered_new_new"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_{model_name}_results.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["nq_serini", "triviaqa", "squad"])
    # parser.add_argument("--datasets", nargs="+", default=["squad"])
    # parser.add_argument("--datasets", nargs="+", default=["nq_serini"])
    # parser.add_argument("--models", nargs="+", default=["gpt4o"])
    # parser.add_argument("--models", nargs="+", default=["gpt4o", "claude35"])
    # parser.add_argument("--models", nargs="+", default=["gpt35", "claude3"])
    parser.add_argument("--models", nargs="+", default=["gpt4o", "ours", "claude35", "gpt35", "claude3"])
    # parser.add_argument("--models", nargs="+", default=["base"])
    args = parser.parse_args()
    
    all_results = {}
    
    for dataset in args.datasets:
        all_results[dataset] = {}
        data_path = f"data/local_index_search/{dataset}/test.parquet"
        
        for model in args.models:
            generations_path = f"results/generations/{dataset}_{model}_generations.json"
            
            if not os.path.exists(generations_path):
                print(f"Skipping {dataset}-{model}: generations file not found")
                continue
                
            print(f"\nProcessing {dataset}-{model}")
            results = process_generations(dataset, model, generations_path, data_path)
            all_results[dataset][model] = results
    
    # Save overall results
    with open("results/answer_filtered_new_new/overall_results_ours.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()


