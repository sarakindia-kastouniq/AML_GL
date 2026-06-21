"""
data_registration.py
====================
Step 1 – Data Registration
- Creates the master project folder structure
- Uploads tourism.csv to Hugging Face dataset hub
"""

import os
import shutil
from huggingface_hub import HfApi, login

# ─────────────────────────────────────────────
# CONFIGURATION  ← update these before running
# ─────────────────────────────────────────────
HF_TOKEN      = os.environ.get("HF_TOKEN", "your_hf_token_here")
HF_USERNAME   = os.environ.get("HF_USERNAME", "your_hf_username_here")
DATASET_REPO  = f"{HF_USERNAME}/tourism-dataset"   # HF dataset repo name
RAW_CSV_PATH  = "tourism.csv"                       # path to the original CSV


def create_folder_structure():
    """Create the master project folder and required sub-folders."""
    folders = [
        "tourism_project",
        "tourism_project/data",
        "tourism_project/model_building",
        "tourism_project/deployment",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"[OK] Folder ready: {folder}")


def copy_data_locally():
    """Copy raw CSV into the project data folder."""
    dest = "tourism_project/data/tourism.csv"
    shutil.copy(RAW_CSV_PATH, dest)
    print(f"[OK] Dataset copied to {dest}")
    return dest


def register_dataset_on_hf(local_csv: str):
    """Create a dataset repository on HF Hub and upload the CSV."""
    login(token=HF_TOKEN)
    api = HfApi()

    # Create the dataset repo (skips if it already exists)
    api.create_repo(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        exist_ok=True,
        private=False,   # keep public so the pipeline can read it without a token
    )
    print(f"[OK] HF dataset repo ready: https://huggingface.co/datasets/{DATASET_REPO}")

    # Upload the raw CSV
    api.upload_file(
        path_or_fileobj=local_csv,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"[OK] tourism.csv uploaded to HF dataset: {DATASET_REPO}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1 – DATA REGISTRATION")
    print("=" * 60)

    create_folder_structure()
    local_csv = copy_data_locally()
    register_dataset_on_hf(local_csv)

    print("\n[DONE] Data registration complete.")
    print(f"       Dataset: https://huggingface.co/datasets/{DATASET_REPO}")
