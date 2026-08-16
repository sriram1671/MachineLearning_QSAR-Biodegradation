# QSAR Biodegradability Lab

### Machine Learning Assignment 2
**R.Sriram Prakash** <br>
**2025ac05101**

Interactive Streamlit app that scores five classical classifiers on the UCI **QSAR Biodegradation** dataset and reports Accuracy, AUC, Precision, Recall, F1 and MCC on test data.

## a. Problem statement

Ready biodegradability is a regulatory property of industrial chemicals. Given 41 quantitative structure–activity relationship (QSAR) molecular descriptors of a chemical, the task is to classify it as **ready biodegradable (RB)** or **not ready biodegradable (NRB)**.

This is a binary classification problem with a moderate class imbalance (about 2:1 NRB:RB). Models are trained on 80% of the chemicals and evaluated on a stratified 20% hold-out test set (`test_data.csv`). Precision, Recall and F1 are reported for the minority class **RB**, which is the scientifically interesting positive class (identifying chemicals that *are* ready biodegradable). AUC uses predicted probability of RB. MCC summarises all four confusion-matrix cells and is robust to imbalance.

## b. Dataset description

| Item | Detail |
| --- | --- |
| Source | UCI Machine Learning Repository / OpenML `qsar-biodeg` version 1 |
| UCI name | QSAR Biodegradation Data Set (Mansouri, Ringsted, Ballabio, Todeschini, Consonni, 2013) |
| Instances | **1055** chemicals (meets ≥ 500) |
| Features | **41** numeric molecular descriptors (meets ≥ 12) |
| Target | `target` ∈ {`NRB`, `RB`} |
| Class counts | NRB = 699 (66.3%), RB = 356 (33.7%) |
| Missing values | None after numeric coercion; a median imputer is still included in every pipeline |
| Split | Stratified 80 / 20, `random_state=42` → 844 train / 211 test rows |

Descriptor families include eigenvalues of graph-theoretic matrices, atom counts (e.g. heavy atoms, halogen counts), charge-related indices and frequency of selected atom pairs. OpenML names them `V1`–`V41`; those names are kept so the uploaded test CSV matches the trained pipelines.

**Why this dataset:** it is large enough for stable metrics, wide enough to satisfy the feature constraint, scientifically interpretable, and less over-used than Titanic / Iris / Pima diabetes, which reduces accidental collision with other submissions.

## c. GitHub repository link

https://github.com/sriram1671/MachineLearning_QSAR-Biodegradation

Repository layout :

```text
ML-ASSIGNMENT-2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── train_models.py          # training + evaluation script
    ├── metrics.json
    ├── feature_names.json
    ├── label_encoder.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest_ensemble.joblib


## d. Models used

All five models required by the assignment comparison table were trained on the **same** train split and scored on the **same** 211-row test set. Logistic Regression, kNN and Gaussian Naive Bayes sit on a median-impute + `StandardScaler` pipeline. Decision Tree and Random Forest use median impute only (no scaling).

Binary Precision / Recall / F1 use **RB as the positive class**.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8436 | 0.9117 | 0.7159 | 0.8873 | 0.7925 | 0.6792 |
| Decision Tree | 0.8294 | 0.8689 | 0.7333 | 0.7746 | 0.7534 | 0.6237 |
| kNN | 0.8483 | 0.9018 | 0.7532 | 0.8169 | 0.7838 | 0.6686 |
| Naive Bayes | 0.6730 | 0.8336 | 0.5076 | 0.9437 | 0.6601 | 0.4680 |
| Random Forest (Ensemble) | **0.8768** | **0.9412** | **0.8358** | 0.7887 | **0.8116** | **0.7208** |



### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong linear baseline (AUC 0.91). Highest Recall among the *calibrated* models (0.89) because `class_weight="balanced"` pushes the decision boundary toward the minority RB class. Precision (0.72) is lower than Random Forest: more false-positive RB calls. The linear separator is a good fit for several additive QSAR indices, but it cannot model descriptor interactions. |
| Decision Tree | Lowest AUC among tree/linear models (0.87) and the weakest MCC except Naive Bayes. `max_depth=10` with leaf constraints reduces memorisation, yet a single tree still fragments the 41-D space into brittle axis-aligned regions. Precision and Recall are reasonably balanced (~0.73–0.77), so errors are not one-sided, but variance is visible versus the forest. |
| kNN | Competitive Accuracy (0.848) and AUC (0.90) after standardisation. Distance-weighted 7-NN recovers local clusters of similar molecules (analogous chemicals biodegrade similarly). It trails Logistic Regression on Recall and Random Forest on every headline metric except being close on Accuracy. Sensitive to irrelevant/correlated descriptors because all 41 features enter the Minkowski distance equally. |
| Naive Bayes | Worst Accuracy (0.67) and MCC (0.47). GaussianNB assumes *conditional independence*; QSAR descriptors are highly collinear (related graph indices). The model becomes over-confident on RB: Recall 0.94 but Precision only 0.51 — it labels many NRB chemicals as ready biodegradable. AUC 0.83 shows ranking ability is not destroyed, but the default 0.5 threshold is poorly calibrated. |
| Random Forest (Ensemble) | **Best overall.** Highest Accuracy, AUC, Precision, F1 and MCC. Bagging plus random feature splits absorbs collinearity and non-linear interactions that hurt NB, kNN and a single tree. Recall (0.79) is slightly below Logistic Regression: the forest is more conservative about calling RB, which *raises* Precision (0.84) and is preferable when a false “ready” label is costly. |
| Overall winner for this dataset? | **Random Forest.** It leads 5 of 6 metrics and has the best MCC (0.72), which is the fairest single number on this imbalanced chemical set. Use Logistic Regression only if you need a more sensitive RB screen (higher Recall) and can tolerate extra false positives. |

## Streamlit app

https://machinelearningqsar-biodegradation.streamlit.app


UI features:

1. CSV upload for **test** data (sidebar), plus a bundled `test_data.csv` checkbox
2. Model selection dropdown (all five classifiers)
3. Live Accuracy / AUC / Precision / Recall / F1 / MCC when `target` is present
4. Confusion matrix heatmap and sklearn classification report
5. Hold-out comparison table so every model’s original test metrics remain visible

