"""
model_building.py
=================
Step 3 – Model Training, Experimentation Tracking & Registration
- Loads train / test splits from Hugging Face dataset hub
- Trains 6 classifiers with hyperparameter tuning (GridSearchCV)
- Logs every experiment to MLflow (parameters, metrics, artefacts)
- Registers the best model to the Hugging Face model hub
"""

import os
import pickle
import tempfile

import mlflow
import mlflow.sklearn
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, login
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# CONFIGURATION  ← update these before running
# ─────────────────────────────────────────────
HF_TOKEN       = os.environ.get("HF_TOKEN", "your_hf_token_here")
HF_USERNAME    = os.environ.get("HF_USERNAME", "your_hf_username_here")
DATASET_REPO   = f"{HF_USERNAME}/tourism-dataset"
MODEL_REPO     = f"{HF_USERNAME}/tourism-wellness-model"   # HF model hub repo
TARGET_COL     = "ProdTaken"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "Tourism_Wellness_Package_Prediction"
RANDOM_STATE    = 42
CV_FOLDS        = 5

# ─────────────────────────────────────────────
# MODELS & HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────
MODELS = {
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "params": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
    },
    "Bagging": {
        "estimator": BaggingClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [10, 50, 100],
            "max_samples": [0.5, 0.8, 1.0],
            "max_features": [0.5, 0.8, 1.0],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        },
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        },
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        ),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    },
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_splits_from_hf():
    """Download train and test CSVs from the HF dataset hub."""
    login(token=HF_TOKEN)
    print("[INFO] Loading train/test splits from Hugging Face …")

    train_ds = load_dataset(DATASET_REPO, data_files="train.csv", split="train")
    test_ds  = load_dataset(DATASET_REPO, data_files="test.csv",  split="train")

    train_df = train_ds.to_pandas()
    test_df  = test_ds.to_pandas()

    print(f"[OK]  Train shape: {train_df.shape}  |  Test shape: {test_df.shape}")
    return train_df, test_df


def evaluate(model, X_test, y_test) -> dict:
    """Return a dict of classification metrics."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_pred_prob), 4),
    }


def train_and_log(name: str, config: dict,
                  X_train, y_train, X_test, y_test) -> tuple:
    """
    Run GridSearchCV, log everything to MLflow, return
    (best_estimator, metrics_dict).
    """
    with mlflow.start_run(run_name=name):

        # ── Hyperparameter search ──────────────────────────────────────
        grid = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["params"],
            cv=CV_FOLDS,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # ── Log best parameters ───────────────────────────────────────
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("model_name", name)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("cv_best_roc_auc", round(grid.best_score_, 4))

        # ── Evaluate on held-out test set ─────────────────────────────
        metrics = evaluate(best_model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # ── Log the model artefact ────────────────────────────────────
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print(f"\n  [{name}]")
        print(f"    Best CV ROC-AUC : {grid.best_score_:.4f}")
        for k, v in metrics.items():
            print(f"    {k:<12}: {v}")

    return best_model, metrics


def register_best_model_on_hf(best_name: str, best_model):
    """Pickle the best model and push it to the HF model hub."""
    api = HfApi()

    # Create model repo (public, exists_ok)
    api.create_repo(
        repo_id=MODEL_REPO,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    print(f"\n[OK]  HF model repo ready: https://huggingface.co/models/{MODEL_REPO}")

    # Save model locally, then upload
    os.makedirs("tourism_project/model_building", exist_ok=True)
    model_path = "tourism_project/model_building/best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.pkl",
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f"[OK]  best_model.pkl ({best_name}) uploaded to HF model hub.")

    # Write a minimal model card
    card = f"""---
license: mit
tags:
  - sklearn
  - classification
  - tourism
  - wellness-package
---

# Tourism Wellness Package – Best Model

**Algorithm:** {best_name}
**Task:** Binary classification – predict whether a customer will purchase the Wellness Tourism Package.
**Dataset:** [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})

## Usage

```python
import pickle
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="{MODEL_REPO}", filename="best_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

prediction = model.predict(X_new)
```
"""
    card_path = "tourism_project/model_building/README.md"
    with open(card_path, "w") as f:
        f.write(card)

    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print("[OK]  Model card (README.md) uploaded.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3 – MODEL TRAINING & REGISTRATION")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    train_df, test_df = load_splits_from_hf()

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test  = test_df.drop(columns=[TARGET_COL])
    y_test  = test_df[TARGET_COL]

    # ── MLflow setup ──────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[OK]  MLflow experiment: '{EXPERIMENT_NAME}'")
    print(f"      Tracking URI      : {MLFLOW_TRACKING_URI}")

    # ── Train all models and collect results ──────────────────────────
    results = {}
    for model_name, model_config in MODELS.items():
        print(f"\n[INFO] Training {model_name} …")
        best_est, metrics = train_and_log(
            model_name, model_config,
            X_train, y_train, X_test, y_test,
        )
        results[model_name] = {"estimator": best_est, "metrics": metrics}

    # ── Select best model by ROC-AUC ──────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best_model = results[best_name]["estimator"]
    best_metrics = results[best_name]["metrics"]

    print("\n" + "=" * 60)
    print(f"BEST MODEL : {best_name}")
    for k, v in best_metrics.items():
        print(f"  {k:<12}: {v}")
    print("=" * 60)

    # Full classification report for the best model
    y_pred = best_model.predict(X_test)
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred))

    # ── Register best model on HF ─────────────────────────────────────
    register_best_model_on_hf(best_name, best_model)

    print("\n[DONE] Model building complete.")
    print(f"       Model hub: https://huggingface.co/{MODEL_REPO}")
