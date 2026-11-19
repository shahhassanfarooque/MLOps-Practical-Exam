import os
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
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
import mlflow
import mlflow.sklearn

# Allow importing dataset_loader from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_loader import DATASET_NAME, get_dataset

def main():
    # 1. Connect to Tracking Server
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    print(f"[*] Setting MLflow Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "exam-classification-baseline"
    mlflow.set_experiment(experiment_name)
    print(f"[*] Active Experiment: '{experiment_name}'")

    # 2. Ingest Active Dataset
    print(f"[*] Loading active dataset: '{DATASET_NAME}'...")
    df, feature_cols, target_col = get_dataset()

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 3. 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Dataset Shape : {df.shape} ({len(feature_cols)} features, {y.nunique()} classes)")
    print(f"  Train samples : {len(X_train)}, Test samples: {len(X_test)}")

    # 4. Define 3 Candidate Models
    # Rule: wrap LogisticRegression inside make_pipeline(StandardScaler()) for convergence safety on digits
    models = {
        "logistic_regression": {
            "estimator": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            ),
            "params": {"model_family": "Linear", "max_iter": 1000, "C": 1.0, "scaler": "StandardScaler"},
        },
        "random_forest_shallow": {
            "estimator": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
            "params": {"model_family": "Ensemble", "n_estimators": 50, "max_depth": 3, "random_state": 42},
        },
        "random_forest_deep": {
            "estimator": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
            "params": {"model_family": "Ensemble", "n_estimators": 100, "max_depth": "None", "random_state": 42},
        },
    }

    results = []

    # 5. Train, Log, and Evaluate Each Candidate
    for model_name, config in models.items():
        print(f"\n========================================================")
        print(f"[*] Training Candidate: '{model_name}'...")
        clf = config["estimator"]
        params = config["params"]

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            # Fit model
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Compute macro metrics
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("dataset_name", DATASET_NAME)
            mlflow.log_param("num_features", len(feature_cols))

            # Log metrics
            mlflow.log_metrics({
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
            })

            # Generate and log Confusion Matrix Artifact
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, ax=ax, cmap="Blues", colorbar=False
            )
            ax.set_title(f"Confusion Matrix - {model_name}\n({DATASET_NAME} dataset)")
            plt.tight_layout()

            cm_path = f"confusion_matrix_{model_name}.png"
            plt.savefig(cm_path)
            plt.close(fig)
            mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
            print(f"  [Artifact] Confusion matrix logged: {cm_path}")

            # Log Model Artifact with input example (using cloudpickle for universal format compatibility)
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                input_example=X_train.iloc[:2],
                serialization_format="cloudpickle",
            )
            print(f"  [Artifact] MLmodel artifact serialized to MLflow run.")

            results.append({
                "Model": model_name,
                "Run ID": run_id[:8] + "...",
                "Accuracy": f"{acc:.4f}",
                "Precision": f"{prec:.4f}",
                "Recall": f"{rec:.4f}",
                "F1-Macro": f"{f1:.4f}",
                "Full_Run_ID": run_id,
                "Raw_F1": f1,
            })

    # 6. Display Comparison Table
    summary_df = pd.DataFrame(results).sort_values(by="Raw_F1", ascending=False)
    print("\n======================= CANDIDATE MODELS COMPARISON =======================")
    print(summary_df[["Model", "Run ID", "Accuracy", "Precision", "Recall", "F1-Macro"]].to_string(index=False))
    print("===========================================================================")
    best_candidate = summary_df.iloc[0]
    print(f"[WINNER] Best Performing Model: '{best_candidate['Model']}' with F1-Macro: {best_candidate['F1-Macro']}")
    print(f"         Run ID: {best_candidate['Full_Run_ID']}")

if __name__ == "__main__":
    main()
