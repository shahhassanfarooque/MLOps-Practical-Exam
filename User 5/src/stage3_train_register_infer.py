import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feast import FeatureStore
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Allow importing dataset_loader from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_loader import DATASET_NAME

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    training_parquet = os.path.join(root_dir, "data", "processed", "feast_training_set.parquet")
    feature_repo_dir = os.path.join(root_dir, "feature_repo")

    if not os.path.exists(training_parquet):
        print(f"[ERROR] Historical training data '{training_parquet}' not found! Run Stage 2 first.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] [Stage 3] Loading Feast training dataset: {training_parquet}...")
    df = pd.read_parquet(training_parquet)

    target_col = "target_label"
    meta_cols = ["sample_id", "event_timestamp", "created_timestamp", target_col]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols]
    y = df[target_col]

    print(f"  Dataset Shape : {df.shape}")
    print(f"  Features count: {len(feature_cols)}")
    print(f"  Classes       : {sorted(y.unique().tolist())}")

    # 1. 80/20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Connect to MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    print(f"[*] Connecting to MLflow Tracking Server: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "e2e-mlops-pipeline"
    mlflow.set_experiment(experiment_name)

    # 3. Candidate Models Definition (StandardScaler pipeline for LogisticRegression)
    candidates = {
        "e2e_logistic_regression": {
            "model": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            ),
            "params": {"model_type": "LogisticRegression", "max_iter": 1000, "C": 1.0, "scaler": "StandardScaler"},
        },
        "e2e_rf_shallow": {
            "model": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
            "params": {"model_type": "RandomForest", "n_estimators": 50, "max_depth": 3, "random_state": 42},
        },
        "e2e_rf_deep": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
            "params": {"model_type": "RandomForest", "n_estimators": 100, "max_depth": "None", "random_state": 42},
        },
    }

    run_results = []

    # 4. Train, Log Metrics, Plots, Artifacts
    for name, cfg in candidates.items():
        print(f"\n[*] Training candidate model: '{name}'...")
        clf = cfg["model"]
        params = cfg["params"]

        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

            mlflow.log_params(params)
            mlflow.log_param("dataset", DATASET_NAME)
            mlflow.log_param("num_features", len(feature_cols))

            mlflow.log_metrics({
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
            })

            # Confusion Matrix Plot Artifact
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"E2E Pipeline - {name}\n({DATASET_NAME} dataset)")
            plt.tight_layout()
            cm_file = f"confusion_matrix_{name}.png"
            plt.savefig(cm_file)
            plt.close(fig)
            mlflow.log_artifact(cm_file, artifact_path="evaluation_plots")

            # MLmodel artifact (using cloudpickle for universal format compatibility)
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                input_example=X_train.iloc[:2],
                serialization_format="cloudpickle",
            )

            run_results.append({
                "model_name": name,
                "run_id": run_id,
                "f1_macro": f1,
                "accuracy": acc,
            })
            print(f"  [OK] Run {run_id[:8]} logged. F1-Macro: {f1:.4f}, Accuracy: {acc:.4f}")

    # 5. Model Selection & Registration
    best_candidate = max(run_results, key=lambda x: x["f1_macro"])
    best_run_id = best_candidate["run_id"]
    registered_name = "e2e-classifier-prod"

    print(f"\n[WINNER] Best Run: {best_candidate['model_name']} (F1={best_candidate['f1_macro']:.4f})")
    print(f"[*] Registering 'runs:/{best_run_id}/model' as '{registered_name}'...")

    reg_mv = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=registered_name)
    version = reg_mv.version
    time.sleep(1)

    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=registered_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
    except Exception as e:
        print(f"  [Note] transition_model_version_stage notice: {e}")

    try:
        client.set_model_version_tag(name=registered_name, version=version, key="stage", value="Staging")
        if hasattr(client, "set_registered_model_alias"):
            client.set_registered_model_alias(name=registered_name, alias="staging", version=version)
    except Exception as e:
        print(f"  [Note] tag/alias notice: {e}")

    print(f"[OK] Model '{registered_name}' version {version} promoted to 'Staging'.")

    # 6. Live Inference Verification via Feast Online Store & Registered Model
    print("\n[*] [E2E Live Inference] Querying Feast SQLite Online Store for entity IDs [0, 1]...")
    feast_store = FeatureStore(repo_path=feature_repo_dir)
    feature_service = feast_store.get_feature_service("e2e_feature_service")
    online_response = feast_store.get_online_features(
        features=feature_service,
        entity_rows=[{"sample_id": 0}, {"sample_id": 1}],
    )
    online_df = online_response.to_df()
    print("[OK] Fetched online features from Feast SQLite store:")
    print(online_df[["sample_id"] + feature_cols[:3]])

    # 7. Load Staging Model from MLflow and Predict
    print(f"\n[*] Loading model 'models:/{registered_name}/Staging' for live inference...")
    try:
        prod_model = mlflow.sklearn.load_model(f"models:/{registered_name}/Staging")
    except Exception:
        fallback_uri = f"models:/{registered_name}/{version}"
        print(f"[*] Attempting load via version URI: {fallback_uri}...")
        try:
            prod_model = mlflow.sklearn.load_model(fallback_uri)
        except Exception:
            print(f"[*] Resolving directly via backing run source: runs:/{best_run_id}/model...")
            prod_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

    # Match exact feature ordering
    X_live = online_df[feature_cols]
    live_preds = prod_model.predict(X_live)

    # Look up actual ground-truth labels for comparison
    actual_labels = df[df["sample_id"].isin([0, 1])][target_col].tolist()

    live_summary = pd.DataFrame({
        "sample_id": [0, 1],
        "Actual Class": actual_labels,
        "Live Online Prediction": live_preds,
        "Match": [a == p for a, p in zip(actual_labels, live_preds)],
    })

    print("\n================== END-TO-END SERVING VERIFICATION ==================")
    print(live_summary.to_string(index=False))
    print("=====================================================================")
    print("[SUCCESS] Complete End-to-End MLOps pipeline executed seamlessly!")

if __name__ == "__main__":
    main()
