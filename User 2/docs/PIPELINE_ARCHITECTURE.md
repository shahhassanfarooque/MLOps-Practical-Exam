# Automated Machine Learning Pipeline Architecture

This document specifies the design, dependency graph, stages, inputs, outputs, and validation rules for the automated DVC pipeline (`part-2-pipeline`).

---

## 1. ASCII Directed Acyclic Graph (DAG)

```
        +-----------------------------------+
        |       dataset_loader.py           |
        +-----------------+-----------------+
                          |
                          v
               +--------------------+
               |   Stage 1: collect  |
               +----------+---------+
                          | data/raw/raw_data.csv
                          v
             +------------------------+
             |   Stage 2: preprocess  |
             +------------+-----------+
                          | data/processed/preprocessed_data.csv
                          v
              +----------------------+
              |   Stage 3: features  |
              +-----------+----------+
                          | data/processed/featured_data.csv
                          |
             +------------+------------+
             |                         |
             v                         v
  +----------------------+  +------------------------+
  |   Stage 4: validate  |  |   Stage 5: train_eval  |
  |  (Quality Hard-Stop) |  |   (RF Model & Metrics) |
  +----------------------+  +------------+-----------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
                       v                                   v
             models/pipeline_model.joblib      metrics/eval_metrics.json
```

---

## 2. Pipeline Stages Specification Table

| Stage | Command | Inputs / Dependencies | Outputs / Artifacts | Purpose & Quality Gate Rules |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: `collect`** | `python src/pipeline/collect.py --output data/raw/raw_data.csv` | `dataset_loader.py`, `src/pipeline/collect.py` | `data/raw/raw_data.csv` | Ingests active runtime dataset (`wine`, `breast_cancer`, or `digits`) and appends UTC ISO metadata timestamp (`collected_at`). |
| **Stage 2: `preprocess`** | `python src/pipeline/preprocess.py --input data/raw/raw_data.csv --output data/processed/preprocessed_data.csv` | `src/pipeline/preprocess.py`, `data/raw/raw_data.csv` | `data/processed/preprocessed_data.csv` | Drops exact duplicate rows; coerces numeric columns; imputes NaNs with median values; drops missing targets; removes timestamp. |
| **Stage 3: `features`** | `python src/pipeline/features.py --input data/processed/preprocessed_data.csv --output data/processed/featured_data.csv` | `src/pipeline/features.py`, `data/processed/preprocessed_data.csv` | `data/processed/featured_data.csv` | Dynamically selects top 2 non-constant numeric features (`col_a`, `col_b`) and builds `feat_interaction_prod`, `feat_interaction_ratio`, and `feat_quantile_bin`. |
| **Stage 4: `validate`** | `python src/pipeline/validate.py --input data/processed/featured_data.csv` | `src/pipeline/validate.py`, `data/processed/featured_data.csv` | None (Terminal Quality Gate) | Validates schema integrity, non-null assertions, classification label cardinality, and absence of infinite/extreme values. Raises `DataValidationError` with hard-stop `sys.exit(1)`. |
| **Stage 5: `train_eval`** | `python src/pipeline/train_eval.py --input data/processed/featured_data.csv --model-out models/pipeline_model.joblib --metrics-out metrics/eval_metrics.json` | `src/pipeline/train_eval.py`, `data/processed/featured_data.csv` | `models/pipeline_model.joblib`, `metrics/eval_metrics.json` | Stratified 80/20 train/test split; fits `RandomForestClassifier(n_estimators=100, random_state=42)`; serializes model and writes macro metrics (`accuracy`, `precision_macro`, `recall_macro`, `f1_macro`). |

---

## 3. Data Validation & Hard-Stop Mechanics

Stage 4 enforces data quality using an explicit hard-stop barrier. If any of the following assertions fail:
1. Target column is missing or improperly named.
2. Any required engineered feature (`feat_interaction_prod`, `feat_interaction_ratio`, `feat_quantile_bin`) is missing.
3. Unexpected `NaN` / `null` values exist in any column.
4. Target label cardinality < 2.
5. `inf` or `-inf` values exist in feature matrices.

The script immediately prints `[VALIDATION FAILURE]` to `stderr` and terminates the execution process with exit code `1`. When running under `dvc repro`, DVC halts the entire DAG pipeline immediately, preventing corrupted data from ever reaching model training.

---

## 4. Caching & Reproducibility Guarantees

DVC computes cryptographic checksums for:
1. Code files (e.g. `src/pipeline/collect.py`).
2. Input datasets.
3. Stage outputs.

When `dvc repro` is executed:
- If neither code nor data dependencies have changed, DVC skips stage execution:
  `Stage 'collect' didn't change, skipping`
- If a dependency changes (e.g. changing `DATASET_NAME` in `dataset_loader.py`), DVC automatically invalidates downstream stages and only re-runs affected nodes.
