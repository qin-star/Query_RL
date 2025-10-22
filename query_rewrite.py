import requests
import json
import argparse
import re
import time

INSTRUCTION = """
You are a query rewriting expert. Your task is to create query terms for user query to find relevant literature in a Wikipedia corpus using BM25.
"""

def format_prompt(user_query: str) -> str:
    """Format the prompt for the model using the same template as make_prefix."""
    input_str = """<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n""" + INSTRUCTION
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<think>
[thinking process]
</think>
<answer>
{
    "query": "...."
} 
</answer>. 
Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately.

Here's the user query:
"""
    input_str += user_query + """
Assistant: Let me rewrite the query with reasoning. 
<think>
"""
    
    return [
        {"role": "system", "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
        {"role": "user", "content": input_str}
    ]

def extract_query(response_text: str) -> str:
    """Extract the rewritten query from the model's response."""
    try:
        # Find the last occurrence of <answer>...</answer>
        if "<answer>" not in response_text:
            response_text = "<answer>" + response_text
        if "</answer>" not in response_text:
            response_text = response_text + "</answer>"
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, response_text, re.DOTALL)
        
        if matches:
            # Get the last matched answer and parse it as JSON
            answer_json = json.loads(matches[-1].strip())
            return answer_json['query']
        else:
            raise ValueError("No answer tags found in response")
    except Exception as e:
        raise ValueError(f"Failed to extract query from response: {e}")

def rewrite_query(query: str, api_url: str = "http://localhost:8000/v1/chat/completions") -> str:
    """Send the query to the vLLM API and get the rewritten version."""
    messages = format_prompt(query)
    
    payload = {
        "model": "DeepRetrieval/DeepRetrieval-NQ-BM25-3B",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract the generated text from the response
        generated_text = result['choices'][0]['message']['content']
        
        # Extract the rewritten query
        rewritten_query = extract_query(generated_text)
        return rewritten_query
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    except Exception as e:
        raise Exception(f"Failed to process response: {e}")

def main():
    parser = argparse.ArgumentParser(description="Query rewriting using vLLM API")
    parser.add_argument("--query", type=str, required=True, help="The query to rewrite")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1/chat/completions",
                      help="URL of the vLLM API server (default: http://localhost:8000/v1/chat/completions)")
    
    args = parser.parse_args()
    
    try:
        start_time = time.time()
        rewritten_query = rewrite_query(args.query, args.api_url)
        end_time = time.time()
        print(f"Original query: {args.query}")
        print(f"Rewritten query: {rewritten_query}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 