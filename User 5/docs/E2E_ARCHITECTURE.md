# End-to-End Unified MLOps Architecture

This document describes the design, architecture, and tool interoperability of the production end-to-end MLOps pipeline (`part-5-e2e-integration`), uniting **Git**, **DVC**, **Feast**, and **MLflow** into a closed-loop system.

---

## 1. Unified Architecture Diagram

```
+===================================================================================================+
|                                    GIT REPOSITORY (Code Versioning)                              |
|                   Tracks pipeline source code, config files, and .dvc pointer files               |
+===================================================================================================+
                                                   |
                                                   v
+---------------------------------------------------------------------------------------------------+
|  STAGE 1: INGESTION, ENGINEERING & VALIDATION (src/stage1_prepare_data.py)                        |
|  - Ingests selected classification dataset via dataset_loader.py (wine / breast_cancer / digits)   |
|  - Deduplicates and imputes missing values                                                        |
|  - Engineers interaction features (prod, ratio)                                                   |
|  - Enforces schema, null, inf, and class cardinality assertions (sys.exit(1) on failure)           |
|  - Generates entity key (sample_id) and UTC timestamps                                            |
|  ==> Output: data/processed/validated_features.parquet                                            |
+---------------------------------------------------------------------------------------------------+
                                                   |
                                                   v
+---------------------------------------------------------------------------------------------------+
|  STAGE 2: FEAST FEATURE STORE (feature_repo/stage2_feast_pipeline.py)                             |
|  - Registers Entity, FileSource, and FeatureViews via `feast apply`                              |
|  - Materializes latest features into SQLite Online Store (data/online_store.db)                   |
|  - Executes point-in-time correct join via `store.get_historical_features()`                      |
|  ==> Output: data/processed/feast_training_set.parquet                                            |
+---------------------------------------------------------------------------------------------------+
                         |                                                   |
                         | (Historical Batch Data)                           | (Low-Latency Point Queries)
                         v                                                   v
+------------------------------------------------+  +-----------------------------------------------+
|  STAGE 3: EXPERIMENT TRACKING & REGISTRATION   |  |  STAGE 3 (Cont.): PRODUCTION ONLINE SERVING   |
|  (src/stage3_train_register_infer.py)          |  |  (src/stage3_train_register_infer.py)         |
|  - Trains 3 candidate models                   |  |  - Fetches fresh entity features from SQLite  |
|  - Logs parameters, metrics & artifacts        |  |    Online Store (get_online_features)         |
|  - Selects best model by macro F1-score        |  |  - Loads registered "Staging" model from      |
|  - Registers in MLflow Registry                |  |    MLflow (models:/e2e-classifier-prod/Staging|
|  - Promotes to "Staging"                       |  |  - Generates verified live real-time labels   |
+------------------------------------------------+  +-----------------------------------------------+
```

---

## 2. Tool Responsibilities Matrix

| Tool | Core Responsibility in E2E Pipeline | Storage / Artifact Mechanism |
| :--- | :--- | :--- |
| **Git** | Codebase version control, team collaboration, Pull Requests | GitHub repository |
| **DVC** | Pipeline execution graph (`dvc repro`), dependency tracking, and stage caching | `dvc.yaml` + `dvc.lock` |
| **Feast** | Central feature definitions, point-in-time historical joins, low-latency online serving | Parquet offline store + SQLite online store (`online_store.db`) |
| **MLflow** | Experiment tracking, hyperparameter logging, confusion matrix artifacts, and model governance | SQLite tracking store (`mlflow.db`) + MLflow Model Registry |

---

## 3. Prevention of Failure Modes

1. **Zero Data Leakage:**
   By utilizing Feast's timestamp-based as-of join (`get_historical_features`), the training dataset strictly contains feature states known at the observation timestamp.
2. **Zero Training-Serving Skew:**
   Both model training and live production serving consume feature vectors governed by the same `e2e_feature_service`. Feature calculation is never duplicated across languages or frameworks.
3. **Hard-Stop Quality Gates:**
   Corrupted, missing, infinite, or degenerate datasets are rejected at Stage 1 before compute resources are spent on training or deployment.
4. **Automated Governance:**
   Models are not manually saved as files in folders; they are evaluated, logged with full input examples and signatures, and transitioned systematically to `Staging` via MLflow's Model Registry API.
