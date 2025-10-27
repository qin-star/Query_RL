import argparse
import os
import logging
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
import glob
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Main categories based on the file structure
MAIN_CATEGORIES = [
    'local_index_search',
    'search_engine',
    'sql',
    'raw_data'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Download DeepRetrieval datasets from Huggingface')
    parser.add_argument('--repo_id', type=str, default='DeepRetrieval/datasets', 
                        help='Huggingface repository ID')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Directory to save downloaded datasets')
    parser.add_argument('--categories', nargs='+', default=['all'],
                        help=f'Specific categories to download: {", ".join(MAIN_CATEGORIES)} or "all"')
    parser.add_argument('--datasets', nargs='+', default=[],
                        help='Specific datasets to download (e.g., fever, hotpotqa)')
    parser.add_argument('--token', type=str, default=None,
                        help='Huggingface API token for private repositories (not required for public repos)')
    parser.add_argument('--list_only', action='store_true',
                        help='Only list available datasets without downloading')
    parser.add_argument('--list_files', action='store_true',
                        help='List all files in the repository')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()

def list_available_files(repo_id, token=None):
    """List all files in the repository."""
    try:
        # For public repositories, token is not needed
        files = list_repo_files(repo_id=repo_id, token=token, repo_type="dataset")
        return files
    except Exception as e:
        logger.error(f"Error listing files from repository: {str(e)}")
        logger.info("If this is a public repository, you can ignore token-related errors.")
        return []

def group_files_by_category(files):
    """Group files by their main category."""
    grouped = {}
    
    for file in files:
        parts = file.split('/')
        if len(parts) > 1:
            category = parts[0]
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(file)
    
    return grouped

def group_files_by_dataset(files, category):
    """Group files by dataset within a category."""
    grouped = {}
    
    for file in files:
        parts = file.split('/')
        if len(parts) > 2 and parts[0] == category:
            dataset = parts[1]
            if dataset not in grouped:
                grouped[dataset] = []
            grouped[dataset].append(file)
    
    return grouped

def download_files(repo_id, files, output_dir, token=None, verbose=False):
    """Download a list of files from the repository."""
    downloaded = 0
    failed = 0
    
    for file in tqdm(files, desc="Downloading files"):
        try:
            # Make sure the output directory exists
            file_dir = os.path.join(output_dir, os.path.dirname(file))
            os.makedirs(file_dir, exist_ok=True)
            
            # Download the file - note: token is optional for public repos
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                repo_type="dataset",
                local_dir=output_dir,
                token=token,
                local_dir_use_symlinks=False
            )
            
            if verbose:
                logger.info(f"Downloaded {file}")
            
            downloaded += 1
        except Exception as e:
            logger.error(f"Error downloading {file}: {str(e)}")
            failed += 1
    
    logger.info(f"Downloaded {downloaded} files. {failed} files failed.")
    return downloaded, failed

def download_category(repo_id, category, output_dir, token=None, datasets=None, verbose=False):
    """Download a specific category from the repository."""
    try:
        # Get all files in the repository
        all_files = list_available_files(repo_id, token)
        
        # Filter files for this category
        category_files = [f for f in all_files if f.startswith(f"{category}/")]
        
        if not category_files:
            logger.warning(f"No files found for category '{category}'")
            return 0
            
        # If specific datasets are requested, filter further
        if datasets:
            filtered_files = []
            for file in category_files:
                parts = file.split('/')
                if len(parts) > 2 and parts[1] in datasets:
                    filtered_files.append(file)
            category_files = filtered_files
        
        if not category_files:
            logger.warning(f"No files match the requested datasets in category '{category}'")
            return 0
            
        # Download the files
        logger.info(f"Downloading {len(category_files)} files for category '{category}'")
        downloaded, _ = download_files(repo_id, category_files, output_dir, token, verbose)
        
        return downloaded
    except Exception as e:
        logger.error(f"Error downloading category '{category}': {str(e)}")
        return 0

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Connecting to repository: {args.repo_id}")
    if args.token:
        logger.info("Using provided API token for authentication")
    else:
        logger.info("No token provided - accessing as public repository")
    
    # Get list of all files in the repository
    all_files = list_available_files(args.repo_id, args.token)
    
    if args.list_files:
        logger.info(f"Files in repository ({len(all_files)}):")
        for file in all_files:
            print(file)
        return
        
    # Group files by category
    grouped_by_category = group_files_by_category(all_files)
    
    if args.list_only:
        if not grouped_by_category:
            logger.warning("No categories found. Is the repository public and correctly specified?")
            return
            
        logger.info(f"Available categories ({len(grouped_by_category)}):")
        for category, files in grouped_by_category.items():
            print(f"{category} ({len(files)} files)")
            
            # Group by dataset within the category
            if category in MAIN_CATEGORIES:
                datasets = group_files_by_dataset(files, category)
                for dataset, dataset_files in datasets.items():
                    print(f"  - {dataset} ({len(dataset_files)} files)")
        return
    
    # Determine which categories to download
    categories_to_download = []
    if 'all' in args.categories:
        categories_to_download = MAIN_CATEGORIES
    else:
        categories_to_download = [cat for cat in args.categories if cat in grouped_by_category]
    
    if not categories_to_download:
        logger.warning("No valid categories to download.")
        return
        
    # Download each category
    total_downloaded = 0
    for category in categories_to_download:
        if category in grouped_by_category:
            logger.info(f"Downloading category: {category}")
            downloaded = download_category(
                args.repo_id, 
                category, 
                args.output_dir, 
                args.token, 
                args.datasets if args.datasets else None,
                args.verbose
            )
            total_downloaded += downloaded
        else:
            logger.warning(f"Category '{category}' not found in repository.")
    
    logger.info(f"Download complete. Total files downloaded: {total_downloaded}")

if __name__ == "__main__":
    main()