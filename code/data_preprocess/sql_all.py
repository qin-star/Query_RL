# generate bird and spider sql data first
import pandas as pd
import random
import os


datasets = ['bird', 'spider']
splits = ['train', 'test']


seed = 42
random.seed(seed)

if not os.path.exists('data/sql/sql_all'):
    os.makedirs('data/sql/sql_all')

for split in splits:
    merged_df = pd.DataFrame()
    
    for dataset in datasets:
        path = f'data/sql/{dataset}/{split}.parquet'
        df = pd.read_parquet(path)
        # print(f"dataset: {dataset}, split: {split}, Loaded {len(df)} rows")
        
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    if split == 'train':
        merged_df = merged_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    merged_path = f'data/sql/sql_all/{split}.parquet'
    merged_df.to_parquet(merged_path)
    print(f"Merged dataset saved at {merged_path} with {len(merged_df)} rows")
