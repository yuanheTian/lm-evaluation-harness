import os

# —— 新增 ——
# 指定 HF 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import (
    get_dataset_config_names,
    load_dataset,
    DownloadMode,
)
import time
import shutil

# Define the dataset and base save directory
DATASET_NAME = "openai/gsm8k"
configs = ['main', 'socratic']
BASE_SAVE_DIR = DATASET_NAME

if os.path.exists(BASE_SAVE_DIR):
    shutil.rmtree(BASE_SAVE_DIR)

# Step 2: load every config into memory
datasets_in_memory = {}
for config in configs:
    print(f"Loading config '{config}' into memory...")
    ds = load_dataset(
        DATASET_NAME,
        config
    )
    datasets_in_memory[config] = ds
    time.sleep(1)

# Step 3: save each loaded dataset to disk
for config, ds in datasets_in_memory.items():
    save_dir = os.path.join(BASE_SAVE_DIR, config)
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    # os.makedirs(save_dir, exist_ok=True)

    try:
        ds.save_to_disk(save_dir)
        print(f"✔️  Saved '{config}' to '{save_dir}'")
    except Exception as e:
        print(f"  ❗ error saving '{config}': {e}")
        shutil.rmtree(save_dir)
