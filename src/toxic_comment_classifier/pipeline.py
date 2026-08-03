"""Training and evaluation pipeline for binary toxic-comment classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


def load_dataset(
    path: str | Path,
    text_column: str = "comment_text",
    target_column: str = "toxic",
) -> pd.DataFrame:
    """Load and validate a CSV, Excel, or JSON binary text dataset."""
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(dataset_path, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(dataset_path)
    elif suffix == ".json":
        frame = pd.read_json(dataset_path)
    else:
        raise ValueError("Unsupported dataset format. Use CSV, XLSX, XLS, or JSON.")

    missing = {text_column, target_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    prepared = frame[[text_column, target_column]].copy()
    prepared[text_column] = prepared[text_column].fillna("").astype(str).str.strip()
    prepared[target_column] = pd.to_numeric(prepared[target_column], errors="coerce")
    prepared = prepared.dropna(subset=[target_column])
    prepared = prepared[prepared[text_column].ne("")]

    numeric_labels = prepared[target_column]
    if not numeric_labels.map(lambda value: float(value).is_integer()).all():
        raise ValueError("Target column must contain integer binary labels 0 and 1 only.")
    prepared[target_column] = numeric_labels.astype(int)
    labels = set(prepared[target_column].unique())
    if not labels or not labels.issubset({0, 1}):
        raise ValueError("Target column must contain binary labels 0 and 1 only.")
    if labels != {0, 1}:
        raise ValueError("Dataset must contain both non-toxic (0) and toxic (1) labels.")
    if len(prepared) < 10:
        raise ValueError("Dataset must contain at least 10 usable comments.")

    class_counts = prepared[target_column].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each target class must contain at least two comments.")

    return prepared.reset_index(drop=True)


def build_pipeline(max_features: int = 50_000, random_state: int = 42) -> Pipeline:
    """Build word/character TF-IDF features with balanced Logistic Regression."""
    if max_features < 100:
        raise ValueError("max_features must be at least 100.")

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    strip_accents="unicode",
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
        solver="liblinear",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def run_experiment(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    text_column: str = "comment_text",
    target_column: str = "toxic",
    test_size: float = 0.2,
    max_features: int = 50_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, and persist a baseline experiment."""
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5.")

    frame = load_dataset(dataset_path, text_column, target_column)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[text_column],
        frame[target_column],
        test_size=test_size,
        random_state=random_state,
        stratify=frame[target_column],
    )
    model = build_pipeline(max_features=max_features, random_state=random_state)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics: dict[str, Any] = {
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "text_column": text_column,
        "target_column": target_column,
        "random_state": random_state,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_toxic": precision_score(y_test, predictions, zero_division=0),
        "recall_toxic": recall_score(y_test, predictions, zero_division=0),
        "f1_toxic": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["non_toxic", "toxic"],
            output_dict=True,
            zero_division=0,
        ),
    }

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.json"
    model_path = destination / "toxic_comment_pipeline.joblib"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    joblib.dump(model, model_path)
    return {**metrics, "metrics_path": str(metrics_path), "model_path": str(model_path)}
