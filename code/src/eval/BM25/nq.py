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
from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from tqdm import tqdm


_searcher = None
_tokenizer = SimpleTokenizer()

def get_searcher():
    """Lazily initialize and return the searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
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


def evaluate_model(model, tokenizer, data_path, device, model_name, save_dir, batch_size=8, search_api=None):
    df = pd.read_parquet(data_path)
    
    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['label'].tolist()
    all_generated_query = []
    generated_text_list = []
    
    model = model.to(device)
    error_count = 0
    
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    recall_at_20 = []
    recall_at_50 = []
    recall_at_100 = []
    
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        
        # Fix for Mistral models
        if "mistral" in model_name.lower():
            # Process each input individually for Mistral
            outputs = []
            for single_input in batch_inputs:
                # Format with Mistral chat template
                if hasattr(tokenizer, 'apply_chat_template'):
                    # Use the built-in chat template
                    messages = [{"role": "user", "content": single_input}]
                    chat_text = tokenizer.apply_chat_template(messages, return_tensors="pt")
                    
                    # Create attention mask (all 1s since there's no padding in a single sequence)
                    attention_mask = torch.ones_like(chat_text)
                    
                    formatted_input = {
                        'input_ids': chat_text.to(device),
                        'attention_mask': attention_mask.to(device)
                    }
                else:
                    # Manual template for older versions
                    encoded = tokenizer(f"<s>[INST] {single_input} [/INST]", 
                                       return_tensors="pt", 
                                       padding=True, 
                                       truncation=True)
                    formatted_input = {k: v.to(device) for k, v in encoded.items()}
                
                with torch.no_grad():
                    # Set pad_token_id to eos_token_id if not already set
                    pad_token_id = tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = tokenizer.eos_token_id
                        
                    output = model.generate(
                        **formatted_input,
                        max_new_tokens=512,
                        pad_token_id=pad_token_id
                    )
                outputs.append(output)
            
            output_ids = outputs
        else:
            # Original method for non-Mistral models
            tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                output_ids = model.generate(**tokenized_inputs, max_new_tokens=512)
        
        for i, output in enumerate(output_ids):
            try:
                if "mistral" in model_name.lower():
                    # For Mistral, use our robust decoding logic
                    try:
                        if hasattr(output, 'shape') and len(output.shape) > 1:
                            # It's a tensor with multiple sequences
                            output_tensor = output[0]
                        elif isinstance(output, list):
                            if isinstance(output[0], list):
                                # It's a list of lists
                                output_tensor = torch.tensor(output[0])
                            elif isinstance(output[0], torch.Tensor):
                                # It's a list of tensors
                                output_tensor = output[0]
                            else:
                                # It's a list of integers
                                output_tensor = torch.tensor(output)
                        else:
                            # It's already a tensor
                            output_tensor = output
                            
                        # Make sure we have a tensor before decoding
                        if not isinstance(output_tensor, torch.Tensor):
                            output_tensor = torch.tensor(output_tensor)
                            
                        generated_text = tokenizer.decode(output_tensor, skip_special_tokens=True)
                    except Exception as e:
                        print(f"Decoding error: {e}")
                        print(f"Output type: {type(output)}")
                        if isinstance(output, list):
                            print(f"First element type: {type(output[0])}")
                        # Fallback to safer decoding
                        if isinstance(output, list) and len(output) > 0:
                            if isinstance(output[0], torch.Tensor):
                                generated_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
                            else:
                                # Handle nested list case
                                flat_ids = []
                                if isinstance(output[0], list):
                                    flat_ids = output[0]
                                else:
                                    flat_ids = output
                                generated_text = tokenizer.decode(flat_ids, skip_special_tokens=True)
                else:
                    # For non-Mistral models
                    generated_text = tokenizer.decode(output, skip_special_tokens=True)
                
                generated_text_list.append(generated_text)
                idx = batch_start + i
                
                extracted_solution, processed_str = extract_solution(generated_text)
                query = json.loads(extracted_solution)['query']
                
                rank = 1001
                target = targets[idx].tolist()
                searched_doc_list = run_index_search_bm25(query, topk=100)
                
                for j in range(len(searched_doc_list)):
                    if has_answers(searched_doc_list[j], target, _tokenizer, regex=False):
                        rank = j + 1
                        break
                
                all_generated_query.append({'generated_query': query, 'rank': rank, 'target': target})
                    
                recall_at_1.append(1) if rank <= 1 else recall_at_1.append(0)
                recall_at_5.append(1) if rank <= 5 else recall_at_5.append(0)
                recall_at_10.append(1) if rank <= 10 else recall_at_10.append(0)
                recall_at_20.append(1) if rank <= 20 else recall_at_20.append(0)
                recall_at_50.append(1) if rank <= 50 else recall_at_50.append(0)
                recall_at_100.append(1) if rank <= 100 else recall_at_100.append(0)
                
            except Exception as e:
                recall_at_1.append(0)
                recall_at_5.append(0)
                recall_at_10.append(0)
                recall_at_20.append(0)
                recall_at_50.append(0)
                recall_at_100.append(0)
                error_count += 1
                print(f"Error: {e}")
                print(f"Generated text: {generated_text[:200]}...")  # Print first 200 chars for debugging
                continue
            
        try:
            print(f"Recall@1: {sum(recall_at_1) / len(recall_at_1)}")
            print(f"Recall@5: {sum(recall_at_5) / len(recall_at_5)}")
            print(f"Recall@20: {sum(recall_at_20) / len(recall_at_20)}")
            print(f"Recall@100: {sum(recall_at_100) / len(recall_at_100)}")
        except:
            continue
    
    with open(os.path.join(save_dir, f"{model_name}_generations.json"), 'w') as f:
        json.dump(generated_text_list, f, indent=2)
    
    print("Error count: ", error_count)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/shared/eng/pj20/lmr_model/nq_serini_3b/actor/global_step_400")
    parser.add_argument("--data_path", type=str, default="data/local_index_search/nq_serini/test.parquet")
    parser.add_argument("--model_name", type=str, default="nq_ours")
    parser.add_argument("--save_dir", type=str, default="results/generations")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path)
    evaluate_model(model, tokenizer, args.data_path, device, args.model_name, args.save_dir, args.batch_size)

if __name__ == "__main__":
    main()
