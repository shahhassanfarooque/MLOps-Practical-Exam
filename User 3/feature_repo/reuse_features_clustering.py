import os
import sys
import pandas as pd
from sklearn.cluster import KMeans
from feast import FeatureStore

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    store = FeatureStore(repo_path=repo_dir)

    parquet_path = os.path.join(repo_dir, "data", "features.parquet")
    if not os.path.exists(parquet_path):
        print(f"[ERROR] Parquet offline file '{parquet_path}' not found! Run setup_feature_store.py first.")
        sys.exit(1)

    print("[*] Demonstrating Feature Reusability across distinct ML tasks...")
    print("    Retrieving features from Feast 'exam_feature_service' for Unsupervised KMeans Clustering")
    print("    (Zero duplicate feature engineering code required!)")

    raw_df = pd.read_parquet(parquet_path)
    entity_df = raw_df[["sample_id", "event_timestamp"]].copy()

    feature_service = store.get_feature_service("exam_feature_service")
    dataset = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service,
    ).to_df()

    # Drop entity key and timestamp to isolate feature matrix
    ignore_cols = ["sample_id", "event_timestamp", "created_timestamp", "target_label"]
    feature_cols = [c for c in dataset.columns if c not in ignore_cols]

    X = dataset[feature_cols].copy()
    print(f"  Feature Matrix Shape: {X.shape} ({len(feature_cols)} features)")

    # Fit unsupervised KMeans model
    n_clusters = 3
    print(f"[*] Fitting KMeans(n_clusters={n_clusters}, random_state=42)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    print("\n[SUCCESS] Unsupervised Clustering Model trained using shared Feast features!")
    print(f"  Cluster Assignments Distribution: {counts}")
    print(f"  Inertia (Sum of squared distances): {kmeans.inertia_:.2f}")

if __name__ == "__main__":
    main()
