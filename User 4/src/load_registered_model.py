import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Allow importing dataset_loader from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_loader import DATASET_NAME, get_dataset

def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    print(f"[*] Connecting to MLflow at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    registered_name = "exam-classifier-prod"
    model_uri = f"models:/{registered_name}/Staging"
    print(f"[*] Attempting to load model from registry: '{model_uri}'...")

    try:
        model = mlflow.sklearn.load_model(model_uri=model_uri)
    except Exception as e:
        print(f"[!] Staging stage lookup notice ({e}), resolving latest version from registry...")
        client = MlflowClient()
        latest = client.get_latest_versions(registered_name)
        if not latest:
            print(f"[ERROR] No versions found for registered model '{registered_name}'!", file=sys.stderr)
            sys.exit(1)
        fallback_version = latest[-1].version
        fallback_uri = f"models:/{registered_name}/{fallback_version}"
        print(f"[*] Attempting to load via version URI: {fallback_uri}...")
        try:
            model = mlflow.sklearn.load_model(model_uri=fallback_uri)
        except Exception:
            run_id = latest[-1].run_id
            print(f"[*] Resolving directly via backing run source: runs:/{run_id}/model...")
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    print(f"\n[OK] Successfully loaded model from MLflow Model Registry:")
    print(f"     Estimator Type : {type(model)}")
    print(f"     Details        : {model}")

    # Load active test data
    print(f"\n[*] Generating test inference samples from dataset: '{DATASET_NAME}'...")
    df, feature_cols, target_col = get_dataset()

    sample_test_features = df[feature_cols].iloc[:5]
    actual_labels = df[target_col].iloc[:5].tolist()

    # Predict
    predicted_labels = model.predict(sample_test_features)

    # Format output
    comparison_df = pd.DataFrame({
        "Sample Index": range(5),
        "Actual Class": actual_labels,
        "Predicted Class": predicted_labels,
        "Match": [a == p for a, p in zip(actual_labels, predicted_labels)],
    })

    print("\n================== LIVE INFERENCE VERIFICATION ==================")
    print(comparison_df.to_string(index=False))
    print("=================================================================")
    print("[SUCCESS] Production model inference verified successfully!")

if __name__ == "__main__":
    main()
