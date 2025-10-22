import argparse
import os
import logging
from huggingface_hub import HfApi, upload_folder, create_repo
import tempfile
import shutil
import glob
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Upload DeepRetrieval datasets to Huggingface')
    parser.add_argument('--input_dir', type=str, default='./data',
                        help='Root directory containing the datasets')
    parser.add_argument('--repo_id', type=str, default='DeepRetrieval/datasets',
                        help='Huggingface repository ID to upload to')
    parser.add_argument('--token', type=str, required=True,
                        help='Huggingface API token')
    parser.add_argument('--upload_categories', nargs='+', 
                        default=['local_index_search', 'search_engine', 'sql', 'raw_data'],
                        help='Categories to upload (subdirectories of input_dir)')
    parser.add_argument('--skip_datasets', nargs='+', default=['old', 'others'],
                        help='Datasets or directories to skip')
    parser.add_argument('--create_new_repo', action='store_true',
                        help='Create a new repository if it doesn\'t exist')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--dry_run', action='store_true',
                        help='List files to be uploaded without actually uploading')
    return parser.parse_args()

def create_huggingface_repo(repo_id, token, repo_type="dataset"):
    """Create a new Huggingface repository if it doesn't exist."""
    try:
        api = HfApi(token=token)
        create_repo(repo_id=repo_id, token=token, repo_type=repo_type, exist_ok=True)
        logger.info(f"Repository '{repo_id}' created/confirmed.")
        return True
    except Exception as e:
        logger.error(f"Failed to create repository '{repo_id}': {str(e)}")
        return False

def should_skip_path(path, skip_patterns):
    """Check if a path should be skipped based on patterns."""
    for pattern in skip_patterns:
        if pattern in path:
            return True
    return False

def get_upload_paths(input_dir, categories, skip_patterns, verbose=False):
    """Get paths of all categories to upload."""
    upload_paths = []
    
    for category in categories:
        category_path = os.path.join(input_dir, category)
        if not os.path.exists(category_path):
            logger.warning(f"Category directory '{category_path}' doesn't exist. Skipping.")
            continue
            
        # Add the category path itself
        upload_paths.append({
            'source_path': category_path,
            'target_path': category
        })
        
        if verbose:
            logger.info(f"Added category '{category}' for upload.")
    
    # Filter out paths that should be skipped
    filtered_paths = []
    for path_info in upload_paths:
        if not should_skip_path(path_info['source_path'], skip_patterns):
            filtered_paths.append(path_info)
        elif verbose:
            logger.info(f"Skipping path '{path_info['source_path']}'")
    
    return filtered_paths

def upload_category(source_path, target_path, repo_id, token, verbose=False, dry_run=False):
    """Upload a category to Huggingface."""
    try:
        logger.info(f"Preparing to upload {source_path} to {repo_id}/{target_path}")
        
        # Create temporary directory with the right structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # Replicate the directory structure in the temp dir
            temp_target_path = os.path.join(temp_dir, target_path)
            os.makedirs(os.path.dirname(temp_target_path), exist_ok=True)
            
            # Copy the files
            if os.path.isdir(source_path):
                shutil.copytree(source_path, temp_target_path)
            else:
                shutil.copy2(source_path, temp_target_path)
                
            # Count files for reporting
            total_files = sum(1 for _ in glob.iglob(os.path.join(temp_dir, '**', '*'), recursive=True) if os.path.isfile(_))
            logger.info(f"Prepared {total_files} files for upload")
            
            if verbose:
                # List some of the files that would be uploaded
                sample_files = list(glob.iglob(os.path.join(temp_dir, '**', '*.parquet'), recursive=True))[:10]
                if sample_files:
                    logger.info(f"Sample files to upload:")
                    for f in sample_files:
                        logger.info(f"  - {os.path.relpath(f, temp_dir)}")
            
            if dry_run:
                logger.info(f"DRY RUN: Would upload {total_files} files to {repo_id}/{target_path}")
                return True
                
            # Upload the files
            upload_folder(
                folder_path=temp_dir,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                ignore_patterns=["*.git*", "*.ipynb_checkpoints*", "*__pycache__*"]
            )
            
            # Log extra information if verbose
            if verbose:
                logger.info(f"Upload to {repo_id} completed")
            
            logger.info(f"Successfully uploaded {total_files} files to {repo_id}/{target_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error uploading '{source_path}': {str(e)}")
        return False

def main():
    args = parse_args()
    
    # Validate the input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory '{args.input_dir}' does not exist.")
        return
    
    # Create the repository if requested
    if args.create_new_repo:
        if not create_huggingface_repo(args.repo_id, args.token):
            logger.error("Failed to create repository. Aborting upload.")
            return
    
    # Get paths to upload
    paths_to_upload = get_upload_paths(args.input_dir, args.upload_categories, args.skip_datasets, args.verbose)
    
    if not paths_to_upload:
        logger.warning("No valid paths found to upload.")
        return
    
    logger.info(f"Found {len(paths_to_upload)} categories to upload.")
    
    # Upload each category
    successful_uploads = 0
    for path_info in tqdm(paths_to_upload, desc="Uploading categories"):
        if upload_category(
            path_info['source_path'], 
            path_info['target_path'], 
            args.repo_id, 
            args.token, 
            args.verbose,
            args.dry_run
        ):
            successful_uploads += 1
    
    if args.dry_run:
        logger.info(f"DRY RUN complete. Would have uploaded {len(paths_to_upload)} categories.")
    else:
        logger.info(f"Upload complete. {successful_uploads}/{len(paths_to_upload)} categories uploaded successfully.")

if __name__ == "__main__":
    main()