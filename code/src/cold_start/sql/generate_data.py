import os
import json
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import traceback

from src.utils.gpt_azure import gpt_chat_4o



dataset = 'bird'
# dataset = 'spider'
# dataset = 'wikisql'


MAX_WORKERS = 5


rl_data_parquet_path = f'data/sql/{dataset}/train.parquet'



df = pd.read_parquet(rl_data_parquet_path)
rl_data = [df.iloc[i] for i in range(len(df))]

print(len(rl_data))

sft_data_path = f'data/sql/cold_start/{dataset}_reason_sft_train-.jsonl'

if os.path.exists(sft_data_path):
    with open(sft_data_path, 'r') as f:
        sft_data = [json.loads(line) for line in f]
else:
    sft_data = [{}] * len(rl_data)



def worker(thread_id, D):
    if sft_data[thread_id] != {}:
        return thread_id, sft_data[thread_id]

    data_id = thread_id
    original_prompt = D['prompt'][0]['content']
    gt_sql = D['sql']

    system_prompt = original_prompt.split('<|im_start|>system')[1].split('<|im_end|>')[0].strip()
    user_prompt = original_prompt.split('<|im_start|>user')[1].split('<|im_end|>')[0].strip()
    assistant_prefix = original_prompt.split('<|im_start|>assistant')[1].strip().split('<think>')[0].strip()
    hint_prompt = f"""The ground truth SQL is: {gt_sql}
You need to give the thinking process and the SQL query structured within <think> and <answer> tags.
"""


    teacher_model_prompt = system_prompt + user_prompt + hint_prompt

    response = gpt_chat_4o(teacher_model_prompt)

    new_prompt = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': user_prompt
        },
        {
            'role': 'assistant',
            'content': assistant_prefix + '\n' + response
        }
    ]
    D = {
        'id': data_id,
        'original_prompt': original_prompt,
        'gt_sql': gt_sql,
        'response': response,
        'new_prompt': new_prompt
    }
    return thread_id, D


error_count = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(worker, i, D) for i, D in enumerate(rl_data)]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            thread_id, D = future.result()
            sft_data[thread_id] = D
        except Exception as e:
            traceback.print_exc()
            error_count += 1
print(f'Error count: {error_count}')




# save the data
with open(sft_data_path, 'w') as f:
    for D in sft_data:
        f.write(json.dumps(D) + '\n')


