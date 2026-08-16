"""
Train five classification models on the UCI QSAR Biodegradation dataset
and persist fitted pipelines, hold-out test data, and evaluation metrics.

Run from the project root:
    python model/train_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

CLASS_ORDER = ["NRB", "RB"]  # NRB=0, RB=1; RB (minority) is the binary positive class
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_ORDER)}

RANDOM_STATE = 42
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load QSAR biodegradation (OpenML / UCI): 41 features, 1055 chemicals."""
    bundle = fetch_openml(name="qsar-biodeg", version=1, as_frame=True, parser="auto")
    X = bundle.data.copy()
    y_raw = bundle.target.astype(str)
    # UCI: 356 ready (RB) and 699 not-ready (NRB). OpenML codes this as 2=RB, 1=NRB.
    y = y_raw.map({"1": "NRB", "2": "RB"}).astype("string")
    X.columns = [str(c) for c in X.columns]
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, y


def build_models() -> dict[str, Pipeline]:
    """Return named sklearn pipelines. Trees skip scaling; others are standardized."""
    numeric_prep_scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_prep_tree = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return {
        "Logistic Regression": Pipeline(
            [
                ("prep", numeric_prep_scaled),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        C=0.75,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("prep", numeric_prep_tree),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=10,
                        min_samples_split=12,
                        min_samples_leaf=6,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "kNN": Pipeline(
            [
                ("prep", numeric_prep_scaled),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                        metric="minkowski",
                        p=2,
                    ),
                ),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("prep", numeric_prep_scaled),
                ("clf", GaussianNB(var_smoothing=1e-8)),
            ]
        ),
        "Random Forest (Ensemble)": Pipeline(
            [
                ("prep", numeric_prep_tree),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=14,
                        min_samples_split=8,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def safe_auc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    """Binary or multiclass AUC (one-vs-rest, macro)."""
    if n_classes == 2:
        if y_proba.ndim == 2:
            scores = y_proba[:, 1]
        else:
            scores = y_proba
        return float(roc_auc_score(y_true, scores))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_classes: int) -> dict:
    average = "binary" if n_classes == 2 else "macro"
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(safe_auc(y_true, y_proba, n_classes), 4),
        "Precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y_raw = load_dataset()
    encoder = {
        "classes": CLASS_ORDER,
        "mapping": CLASS_TO_ID,
        "meaning": {
            "NRB": "Not ready biodegradable (OpenML class 1, 699 chemicals)",
            "RB": "Ready biodegradable (OpenML class 2, 356 chemicals)",
        },
    }
    y = np.array([CLASS_TO_ID[str(v)] for v in y_raw], dtype=int)
    n_classes = len(CLASS_ORDER)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_frame = X_test.copy()
    test_frame["target"] = [CLASS_ORDER[i] for i in y_test]
    test_frame.to_csv(TEST_DATA_PATH, index=False)

    models = build_models()
    metrics = {
        "dataset": {
            "name": "UCI QSAR Biodegradation (OpenML qsar-biodeg v1)",
            "n_features": int(X.shape[1]),
            "n_instances": int(X.shape[0]),
            "n_classes": n_classes,
            "class_names": CLASS_ORDER,
            "class_meaning": {
                "NRB": "Not ready biodegradable (OpenML class 1)",
                "RB": "Ready biodegradable (OpenML class 2)",
            },
            "target_column": "target",
            "positive_class": "RB",
            "class_counts": {name: int((y_raw == name).sum()) for name in CLASS_ORDER},
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "models": {},
    }

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        scores = evaluate(y_test, y_pred, y_proba, n_classes)
        metrics["models"][name] = scores

        out_path = MODEL_DIR / f"{slug(name)}.joblib"
        joblib.dump(pipeline, out_path)
        print(f"{name:28s}  {scores}")

    joblib.dump(encoder, LABEL_ENCODER_PATH)
    FEATURE_NAMES_PATH.write_text(json.dumps(list(X.columns), indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved test data -> {TEST_DATA_PATH}")
    print(f"Saved metrics    -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
