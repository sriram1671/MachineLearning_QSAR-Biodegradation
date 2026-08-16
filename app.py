"""
QSAR Biodegradability Lab — Streamlit frontend for Assignment 2.

Upload the hold-out test CSV (or any CSV with the same feature columns),
pick a trained classifier, and inspect metrics plus a confusion matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


@st.cache_resource
def load_artifacts():
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text())
    stored_metrics = json.loads(METRICS_PATH.read_text())
    pipelines = {
        name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()
    }
    return encoder, feature_names, stored_metrics, pipelines


def class_names_from(encoder: dict) -> list[str]:
    return [str(c) for c in encoder["classes"]]


def encode_labels(encoder: dict, labels) -> np.ndarray:
    mapping = encoder["mapping"]
    values = pd.Series(labels).astype(str).replace({"1": "NRB", "2": "RB"})
    unknown = sorted(set(values.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown labels {unknown}; expected {list(mapping)}")
    return values.map(mapping).to_numpy(dtype=int)


def decode_labels(encoder: dict, encoded) -> np.ndarray:
    classes = np.array(encoder["classes"])
    return classes[np.asarray(encoded).astype(int)]


def find_target_column(columns: list[str], expected: str | None) -> str | None:
    candidates = []
    if expected:
        candidates.append(expected)
    candidates.extend(["target", "class", "Class", "y", "label", "experimental_class"])
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def compute_auc(y_true, y_proba, n_classes: int) -> float:
    if n_classes == 2:
        scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        return float(roc_auc_score(y_true, scores))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))


def live_metrics(y_true, y_pred, y_proba, n_classes: int) -> dict[str, float]:
    average = "binary" if n_classes == 2 else "macro"
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(compute_auc(y_true, y_proba, n_classes), 4),
        "Precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def render_confusion(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.4,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix on uploaded / selected test data")
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="QSAR Biodegradability Lab",
        page_icon="🧪",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #0f1c24 0%, #16262f 40%, #1c3038 100%); }
        h1, h2, h3, p, label, .stMarkdown { color: #e8f1f2; }
        div[data-testid="stMetricValue"] { color: #7fd1c7; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    encoder, feature_names, stored, pipelines = load_artifacts()
    class_names = class_names_from(encoder)
    n_classes = len(class_names)

    st.title("QSAR Biodegradability Lab")
    st.caption(
        "UCI QSAR Biodegradation · 41 molecular descriptors · binary ready / not-ready classification"
    )

    with st.sidebar:
        st.header("Experiment controls")
        st.write(
            "Upload **test-only** CSV (Streamlit free tier). "
            "The file should contain the 41 descriptor columns. "
            "If a `target` column is present, live metrics are computed."
        )
        uploaded = st.file_uploader("Upload test CSV", type=["csv"])
        use_bundled = st.checkbox("Use bundled hold-out test_data.csv", value=uploaded is None)
        model_name = st.selectbox("Select classifier", list(MODEL_FILES.keys()))
        st.divider()
        st.markdown("**Dataset snapshot**")
        info = stored["dataset"]
        st.write(f"Features: **{info['n_features']}**")
        st.write(f"Instances: **{info['n_instances']}**")
        st.write(f"Classes: **{', '.join(info['class_names'])}**")
        st.write(f"Hold-out: **{int(info['test_size'] * 100)}%** · seed `{info['random_state']}`")

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        source_label = "uploaded file"
    elif use_bundled:
        bundled = ROOT / "test_data.csv"
        if not bundled.exists():
            st.error("Bundled test_data.csv is missing. Re-run `python model/train_models.py`.")
            return
        raw = pd.read_csv(bundled)
        source_label = "bundled hold-out test set"
    else:
        st.info("Upload a CSV or enable the bundled test set in the sidebar.")
        return

    target_col = find_target_column(list(raw.columns), stored["dataset"].get("target_column"))
    work = raw.copy()
    y_true_labels = None
    if target_col:
        y_true_labels = work[target_col]
        work = work.drop(columns=[target_col])

    missing = [c for c in feature_names if c not in work.columns]
    extra = [c for c in work.columns if c not in feature_names]
    if missing:
        st.error(
            "The CSV is missing required descriptor columns: "
            + ", ".join(missing[:12])
            + (" …" if len(missing) > 12 else "")
        )
        return
    X = work[feature_names]
    if extra:
        st.warning(f"Ignored extra columns: {', '.join(extra)}")

    pipeline = pipelines[model_name]
    y_pred_enc = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    y_pred_labels = decode_labels(encoder, y_pred_enc)

    st.subheader(f"Results · {model_name}")
    st.write(f"Scored **{len(X)}** rows from the {source_label}.")

    col_a, col_b = st.columns([1.15, 1])

    with col_a:
        st.markdown("#### Evaluation metrics")
        if y_true_labels is not None:
            try:
                y_true_enc = encode_labels(encoder, y_true_labels)
            except ValueError:
                st.error(
                    "Could not map the target column onto the trained classes "
                    f"{class_names}. Use labels NRB / RB (or keep the bundled test file)."
                )
                return

            scores = live_metrics(y_true_enc, y_pred_enc, y_proba, n_classes)
            m1, m2, m3 = st.columns(3)
            m4, m5, m6 = st.columns(3)
            m1.metric("Accuracy", f"{scores['Accuracy']:.4f}")
            m2.metric("AUC", f"{scores['AUC']:.4f}")
            m3.metric("Precision", f"{scores['Precision']:.4f}")
            m4.metric("Recall", f"{scores['Recall']:.4f}")
            m5.metric("F1", f"{scores['F1']:.4f}")
            m6.metric("MCC", f"{scores['MCC']:.4f}")

            st.markdown("#### Confusion matrix")
            st.pyplot(render_confusion(y_true_enc, y_pred_enc, class_names), clear_figure=True)

            st.markdown("#### Classification report")
            report = classification_report(
                y_true_enc,
                y_pred_enc,
                target_names=class_names,
                zero_division=0,
                digits=4,
            )
            st.code(report, language="text")
        else:
            st.info(
                "No target column found, so metrics cannot be computed. "
                "Showing class predictions only."
            )
            st.markdown("##### Metrics recorded on the original hold-out set")
            recorded = stored["models"][model_name]
            m1, m2, m3 = st.columns(3)
            m4, m5, m6 = st.columns(3)
            m1.metric("Accuracy", f"{recorded['Accuracy']:.4f}")
            m2.metric("AUC", f"{recorded['AUC']:.4f}")
            m3.metric("Precision", f"{recorded['Precision']:.4f}")
            m4.metric("Recall", f"{recorded['Recall']:.4f}")
            m5.metric("F1", f"{recorded['F1']:.4f}")
            m6.metric("MCC", f"{recorded['MCC']:.4f}")

    with col_b:
        st.markdown("#### Comparison on original hold-out test set")
        table = pd.DataFrame(stored["models"]).T
        st.dataframe(table.style.highlight_max(axis=0, color="#1f6f64"), use_container_width=True)

        pred_view = X.copy()
        pred_view.insert(0, "predicted_class", y_pred_labels)
        if n_classes == 2:
            pred_view.insert(1, f"P({class_names[1]})", np.round(y_proba[:, 1], 4))
        if y_true_labels is not None:
            pred_view.insert(0, "true_class", y_true_labels.values)
        st.markdown("#### Row-level predictions")
        st.dataframe(pred_view.head(40), use_container_width=True, height=360)

        csv_bytes = pred_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions CSV",
            data=csv_bytes,
            file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
