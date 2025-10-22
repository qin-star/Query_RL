import random
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import (
    SFTConfig,
    SFTTrainer,
)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'



dataset = 'bird'
# dataset = 'spider'


# wo_reasoning = False
wo_reasoning = True

if wo_reasoning:
    train_epoch = 2
else:
    train_epoch = 4

# for_cold_start = True
# cold_start_data_size = 3000
# for_cold_start = False


model_name = 'Qwen/Qwen2.5-3B-Instruct'



################
# Dataset
################

train_data = []
with open(f'data/sql/cold_start/{dataset}_reason_sft_train.jsonl', 'r') as f:
    for line in f:
        D = json.loads(line)
        if D != {}:
            if not wo_reasoning:
                train_data.append(D['new_prompt'])
            else:
                prompt_wo_reasoning = D['new_prompt']
                prompt_wo_reasoning[0]['content'] = prompt_wo_reasoning[0]['content'].replace("You first think about the reasoning process in the mind and then provides the user with the answer.", "You need to provide the user with the answer.")
                prompt_wo_reasoning[1]['content'] = prompt_wo_reasoning[1]['content'].replace("Show your work in <think> </think> tags. ", "")
                prompt_wo_reasoning[1]['content'] = prompt_wo_reasoning[1]['content'].replace("<think>\n[thinking process]\n</think>", "")
                prompt_wo_reasoning[2]['content'] = '<answer>' + prompt_wo_reasoning[2]['content'].split('<answer>')[1].strip()

                train_data.append(prompt_wo_reasoning)


random.seed(42)
# if for_cold_start:
    # train_data = random.sample(train_data, cold_start_data_size)


train_dataset = Dataset.from_dict({
    "source": [f'{dataset}_reason_sft_train' for _ in range(len(train_data))],
    "messages": train_data,
})


################
# Model
################

model_kwargs = dict(
    trust_remote_code=True,
    attn_implementation='flash_attention_2',
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map='auto',
)
 
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



################
# Training
################
if wo_reasoning:
    output_dir = f"/dev/ana/sft_qwen_7/{dataset}_wo_reasoning"
else:
    output_dir = f"/dev/ana/sft_qwen_7/{dataset}"

training_args = SFTConfig(
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=train_epoch,
    save_strategy='epoch',
    output_dir=output_dir,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    processing_class=tokenizer,
)

trainer.train()


# from huggingface_hub import HfApi

# # Initialize the API
# api = HfApi(token="")
# # Create repository
# repo_id = "windszzlang/DeepRetrieval-SQL"
# api.create_repo(repo_id, exist_ok=True)

# # Upload the pickle file
# api.upload_folder(
#     folder_path=f"/shared/eng/pj20/lmr_model/cold_start/{dataset}",
#     path_in_repo=f"cold_start/{dataset}",
#     repo_id=repo_id
# )