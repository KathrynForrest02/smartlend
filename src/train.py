"""
SmartLend Training Pipeline
Week 4 - Model Factory I

Runs experiments defined in config/train_config.yaml.
Logs parameters and metrics to logs/experiments.csv.
Saves the best model artefact to models/.
"""

import os
import csv
import json
import pickle
import yaml
import logging
from datetime import datetime, UTC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    """Load processed dataset and split into features and target."""
    df = pd.read_csv(config["data"]["processed_path"])

    target = config["data"]["target_column"]

    X = df.drop(
        columns=[target] +
        config["features"].get("drop_columns", [])
    )

    y = df[target]

    logger.info(
        f"Loaded dataset: {X.shape[0]} rows, {X.shape[1]} features"
    )
    logger.info(
        f"Class distribution:\n"
        f"{y.value_counts(normalize=True).round(3)}"
    )

    return X, y


def build_pipeline(model_name: str, model_params: dict):
    """
    Build a scikit-learn Pipeline for the given model.
    """

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if model_name == "logistic_regression":
        classifier = LogisticRegression(**model_params)

    elif model_name == "random_forest":
        classifier = RandomForestClassifier(**model_params)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", scaler),
            ("classifier", classifier),
        ]
    )

    return pipeline


def evaluate_model(
    pipeline,
    X_test,
    y_test,
    model_name: str
):
    """Evaluate a fitted model."""

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"--- {model_name} ---")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(
        f"\n{classification_report(y_test, y_pred)}"
    )

    return {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }


def log_experiment(
    log_path: str,
    params: dict,
    metrics: dict,
    config: dict
):
    """Append experiment details to CSV."""

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_exists = os.path.isfile(log_path)

    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_name": metrics["model_name"],
        "test_size": config["data"]["test_size"],
        "random_state": config["data"]["random_state"],
        "roc_auc": round(metrics["roc_auc"], 4),
        "pr_auc": round(metrics["pr_auc"], 4),
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "parameters_json": json.dumps(
            params,
            sort_keys=True
        ),
    }

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    logger.info(f"Logged experiment to {log_path}")


def save_model(
    pipeline,
    model_name: str,
    models_dir: str
):
    """Save model to disk."""

    os.makedirs(models_dir, exist_ok=True)

    path = os.path.join(
        models_dir,
        f"{model_name}.pkl"
    )

    with open(path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"Saved model to {path}")

    return path


def main():

    config = load_config(
        "config/train_config.yaml"
    )

    X, y = load_data(config)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    logger.info(
        f"Train size: {len(X_train)}, "
        f"Test size: {len(X_test)}"
    )

    results = []

    for model_def in config["models"]:

        model_name = model_def["name"]
        model_params = model_def.get("params", {})

        logger.info(f"Training: {model_name}")

        pipeline = build_pipeline(
            model_name,
            model_params
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
            model_name
        )

        log_experiment(
            config["output"]["log_file"],
            model_params,
            metrics,
            config
        )

        save_model(
            pipeline,
            model_name,
            config["output"]["models_dir"]
        )

        results.append(
            (
                model_name,
                metrics["roc_auc"],
                pipeline
            )
        )

    best_name, best_score, best_pipeline = max(
        results,
        key=lambda r: r[1]
    )

    logger.info(
        f"Best model: {best_name} "
        f"(ROC-AUC: {best_score:.4f})"
    )

    save_model(
        best_pipeline,
        "best_model",
        config["output"]["models_dir"]
    )


if __name__ == "__main__":
    main()