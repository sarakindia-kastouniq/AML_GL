"""
hosting.py
==========
Step 4 – Hosting / Deployment
Pushes all deployment files (app.py, requirements.txt, Dockerfile)
to a Hugging Face Space so the Streamlit app is publicly accessible.
"""

import os
import shutil
from huggingface_hub import HfApi, login

# ─────────────────────────────────────────────
# CONFIGURATION  ← update before running
# ─────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", "your_hf_token_here")
HF_USERNAME = os.environ.get("HF_USERNAME", "your_hf_username_here")
SPACE_REPO  = f"{HF_USERNAME}/tourism-wellness-predictor"   # HF Space repo name
SDK         = "streamlit"                                    # Space SDK

# Files to push (local path → path inside the Space repo)
DEPLOY_FILES = {
    "app.py":          "app.py",
    "requirements.txt": "requirements.txt",
    "tourism_project/deployment/Dockerfile": "Dockerfile",
}


def create_hf_space(api: HfApi):
    """Create the Space repo on HF Hub (public, Streamlit SDK)."""
    api.create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk=SDK,
        exist_ok=True,
        private=False,   # must be public for the grader
    )
    print(f"[OK]  HF Space ready: https://huggingface.co/spaces/{SPACE_REPO}")


def write_space_metadata():
    """
    Write a README.md with the YAML front-matter that HF Spaces
    needs to configure the app (title, SDK, python version, etc.).
    """
    readme = f"""---
title: Tourism Wellness Package Predictor
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
license: mit
---

# Tourism Wellness Package Predictor

Predict whether a customer is likely to purchase the **Wellness Tourism Package**.

Built as part of the *Advanced Machine Learning and MLOps* project.
"""
    with open("README_space.md", "w") as f:
        f.write(readme)
    return "README_space.md"


def push_files_to_space(api: HfApi, readme_path: str):
    """Upload every deployment file to the HF Space."""
    # Push README / metadata
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=SPACE_REPO,
        repo_type="space",
    )
    print("[OK]  README.md pushed to Space.")

    # Push deployment files
    for local_path, repo_path in DEPLOY_FILES.items():
        if not os.path.exists(local_path):
            print(f"[WARN] {local_path} not found – skipping.")
            continue
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=SPACE_REPO,
            repo_type="space",
        )
        print(f"[OK]  {local_path}  →  {repo_path}")


def cleanup(readme_path: str):
    """Remove temporary files created during this script."""
    if os.path.exists(readme_path):
        os.remove(readme_path)


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4 – HOSTING: PUSH TO HUGGING FACE SPACE")
    print("=" * 60)

    login(token=HF_TOKEN)
    api = HfApi()

    create_hf_space(api)
    readme_path = write_space_metadata()
    push_files_to_space(api, readme_path)
    cleanup(readme_path)

    print("\n[DONE] Deployment complete.")
    print(f"       App live at: https://huggingface.co/spaces/{SPACE_REPO}")
