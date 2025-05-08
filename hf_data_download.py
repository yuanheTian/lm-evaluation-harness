from datasets import load_dataset, get_dataset_config_names, DownloadMode
import os
import time
import shutil

# Define the dataset name and the local base directory to save
DATASET_NAME = "cais/mmlu"
BASE_SAVE_DIR = os.path.join("./data", DATASET_NAME)

# Ensure base directory exists
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# Retrieve all configurations (subdirectories) of the dataset
configs = get_dataset_config_names(DATASET_NAME)

for config in configs:
    # Construct the local path matching Hugging Face structure
    save_dir = os.path.join(BASE_SAVE_DIR, config)

    # If directory exists and is non-empty, skip download
    if os.path.isdir(save_dir) and os.listdir(save_dir):
        print(f"Config '{config}' already downloaded at '{save_dir}', skipping.")
        continue

    # Otherwise, prepare directory
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Wait for 1 second before processing next config
    time.sleep(1)

    # Attempt to load dataset, retry with force redownload on failure
    try:
        dataset = load_dataset(DATASET_NAME, config)
    except Exception as e:
        print(f"Error loading config '{config}': {e}. Removing directory and retrying with force download.")
        shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        dataset = load_dataset(DATASET_NAME, config, download_mode=DownloadMode.FORCE_REDOWNLOAD)

    # Save the dataset to disk
    try:
        dataset.save_to_disk(save_dir)
        print(f"Saved config '{config}' to '{save_dir}'")
    except Exception as e:
        print(f"Error saving config '{config}' to '{save_dir}': {e}. Removing directory.")
        shutil.rmtree(save_dir)
