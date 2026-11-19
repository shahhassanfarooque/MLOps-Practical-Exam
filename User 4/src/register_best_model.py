import os
import sys
import time
import mlflow
from mlflow.tracking import MlflowClient

def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    print(f"[*] Connecting to MLflow Client at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment_name = "exam-classification-baseline"
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"[ERROR] Experiment '{experiment_name}' not found! Run train_with_mlflow.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Querying runs in experiment '{experiment_name}' ordered by metrics.f1_macro DESC...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_macro DESC"],
        max_results=5,
    )

    if not runs:
        print("[ERROR] No runs found in experiment!", file=sys.stderr)
        sys.exit(1)

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics.get("f1_macro", 0.0)
    best_model_name = best_run.data.tags.get("mlflow.runName", "unknown")

    print(f"[OK] Found Best Run:")
    print(f"  Run ID       : {best_run_id}")
    print(f"  Model Type   : {best_model_name}")
    print(f"  F1-Macro     : {best_f1:.4f}")

    # Register best model in Model Registry
    registered_model_name = "exam-classifier-prod"
    model_source = f"runs:/{best_run_id}/model"
    print(f"[*] Registering '{model_source}' as model: '{registered_model_name}'...")

    mv = mlflow.register_model(model_uri=model_source, name=registered_model_name)
    version = mv.version
    print(f"[OK] Model registered as Version: {version}")

    # Give MLflow registry metadata a moment to synchronize
    time.sleep(1)

    # Transition to Staging
    print(f"[*] Promoting Version {version} to stage 'Staging'...")
    try:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
    except Exception as e:
        print(f"  [Note] transition_model_version_stage notice: {e}")

    # Also set tag and alias for universal MLflow 2.x and 3.x compatibility
    try:
        client.set_model_version_tag(
            name=registered_model_name,
            version=version,
            key="stage",
            value="Staging",
        )
        if hasattr(client, "set_registered_model_alias"):
            client.set_registered_model_alias(
                name=registered_model_name,
                alias="staging",
                version=version,
            )
    except Exception as e:
        print(f"  [Note] alias/tag notice: {e}")

    print(f"[SUCCESS] Model '{registered_model_name}' (Version {version}) successfully promoted to 'Staging'!")

if __name__ == "__main__":
    main()
