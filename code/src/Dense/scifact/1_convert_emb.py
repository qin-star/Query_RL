import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys
sys.path.append('code')
import pdb
import os

from src.Dense.utils import average_pool



def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, device_map='auto')

def get_dense_embedding(texts):
    batch_dict = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
    return embeddings.cpu().numpy()


def process_corpus(jsonl_path, output_dir):

    doc_ids = []
    texts = []

    embedding_output = f"{output_dir}/embeddings.npy"
    doc_ids_output = f"{output_dir}/doc_ids.npy"

    # Read JSONL corpus
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading corpus"):
            data = json.loads(line.strip())
            doc_ids.append(data["_id"])
            texts.append(data["title"] + " " + data["text"])  # Combine title and text

    # Compute embeddings
    embeddings = []
    batch_size = 8  # Adjust batch size as needed
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding embeddings"):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = get_dense_embedding(batch_texts)
        embeddings.append(batch_embeddings)
    
    # Stack all embeddings
    embeddings = np.vstack(embeddings)

    # Save embeddings and document IDs
    np.save(embedding_output, embeddings)
    np.save(doc_ids_output, np.array(doc_ids))
    
    print(f"Embeddings saved to {embedding_output}, Doc IDs saved to {doc_ids_output}")


ori_data_dir = "code/data/raw_data/scifact/corpus.jsonl"
output_dir = "code/data/local_index_search/scifact/dense"

os.makedirs(output_dir, exist_ok=True)

process_corpus(ori_data_dir, output_dir)