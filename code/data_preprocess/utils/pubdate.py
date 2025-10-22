import sys
import os
import json
from datetime import datetime
import re
from tqdm import tqdm
from verl.utils.apis.pubmed import PubmedAPI
from verl.utils.apis.ctgov import CTGovAPI

# Debug: Print current directory and Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

def convert_date(date_str):
    # Default date if the input is empty or invalid
    default_date = "2025/02/12" # today

    if not date_str.strip():
        return default_date

    # Match patterns
    match = re.match(r'(\d{4})(?: (\w{3}))?(?: (\d{1,2}))?', date_str)
    if not match:
        return default_date

    year = match.group(1)
    month = match.group(2) if match.group(2) else '01'
    day = match.group(3) if match.group(3) else '01'

    # Convert month name to number
    if not month.isdigit():
        try:
            month = datetime.strptime(month, '%b').strftime('%m')
        except ValueError:
            return default_date

    # Format the date string to YYYY/MM/DD
    return f"{year}/{month.zfill(2)}/{day.zfill(2)}"

def save_checkpoint(data, output_file, checkpoint_file):
    # Save the current progress
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    # Save checkpoint info
    checkpoint_info = {
        'processed_count': len(data),
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_info, f)

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def get_pub_date(data, output_file, checkpoint_file, pubmed_api):
    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_file)
    start_idx = checkpoint['processed_count'] if checkpoint else 0
    
    # Load existing processed data if checkpoint exists
    if checkpoint and os.path.exists(output_file):
        processed_data = []
        with open(output_file, 'r') as f:
            for line in f:
                processed_data.append(json.loads(line))
        data = processed_data + data[start_idx:]
    
    try:
        for i, item in enumerate(tqdm(data[start_idx:], initial=start_idx)):
            if 'pub_date' in item.keys():
                continue
            
            try:
                pmid = item['origin']['pmid']
                pub_date = pubmed_api.get_papers_by_pmids([pmid])['Publication Date'].fillna('').tolist()[0]
                item['pub_date'] = convert_date(pub_date)
                
                # Save checkpoint every 100 items
                if (i + 1) % 100 == 0:
                    save_checkpoint(data[:i+1], output_file, checkpoint_file)
                    
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                continue
                
        # Save final results
        save_checkpoint(data, output_file, checkpoint_file)
        return data
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        # Save progress before raising the error
        save_checkpoint(data[:i], output_file, checkpoint_file)
        raise

if __name__ == '__main__':
    # Initialize APIs
    if os.path.exists('verl/utils/reward_score/apis/pubmed_api.key'):
        api_key = open('verl/utils/reward_score/apis/pubmed_api.key', 'r').read().strip()
        pubmed_api = PubmedAPI(api_key=api_key)
    ctgov_api = CTGovAPI()

    # Define data paths
    data_paths = {
        'train': '/home/pj20/server-04/LMR/code/data/raw_data/ctgov/train.jsonl',
        'test': '/home/pj20/server-04/LMR/code/data/raw_data/ctgov/test.jsonl',
        'dev': '/home/pj20/server-04/LMR/code/data/raw_data/ctgov/dev.jsonl'
    }

    for split_name, data_path in data_paths.items():
        print(f"\nProcessing {split_name} split...")
        
        # Load data
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Define checkpoint file
        checkpoint_file = f"{data_path}.checkpoint"
        
        try:
            # Process data with checkpointing
            processed_data = get_pub_date(data, data_path, checkpoint_file, pubmed_api)
            print(f"Successfully processed {split_name} split")
            
            # Remove checkpoint file after successful completion
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                
        except Exception as e:
            print(f"Error processing {split_name} split: {str(e)}")
            print("Progress has been saved. You can resume by running the script again.")
    
    

