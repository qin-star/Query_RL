import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='msmarco_beir')
args = parser.parse_args()

dataset_name = args.dataset_name

file_path = f'../results_dense/no_reason/gpt-4o_{dataset_name}.json'
with open(file_path, 'r') as file:
    data = json.load(file)


answers = {}
for key, value in data.items():
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", value["generated_text"], re.DOTALL)
    try:
        query_text = match.group(1)
        # json loads
        gen_query = json.loads(query_text)['query']   
        answers[key] = {
            "generated_text": gen_query,
            "target": value["target"],
        }
    except:
        answers[key] = {
            "generated_text": value["generated_text"],
            "target": value["target"],
        }


# save to filename gpt-4o_postprocessed_scifact.json
file_path = f'../results_dense/no_reason/gpt-4o_post_{dataset_name}.json'
with open(file_path, 'w') as file:
    json.dump(answers, file, indent=4)

