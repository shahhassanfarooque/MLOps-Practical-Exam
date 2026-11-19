import os
import sys
import pandas as pd
from feast import FeatureStore

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    store = FeatureStore(repo_path=repo_dir)

    parquet_path = os.path.join(repo_dir, "data", "features.parquet")
    if not os.path.exists(parquet_path):
        print(f"[ERROR] Parquet offline file '{parquet_path}' not found! Run setup_feature_store.py first.")
        sys.exit(1)

    print("[*] Loading entity observation frame for point-in-time correct historical retrieval...")
    raw_df = pd.read_parquet(parquet_path)
    entity_df = raw_df[["sample_id", "event_timestamp"]].copy()

    print(f"  Entity rows count: {len(entity_df)}")
    print("[*] Executing store.get_historical_features() using 'exam_feature_service'...")

    feature_service = store.get_feature_service("exam_feature_service")
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service,
    )

    training_df = training_data.to_df()

    print("\n[SUCCESS] Retrieved Historical Training Set (Point-in-Time Correct):")
    print(f"  Training DataFrame shape : {training_df.shape}")
    print(f"  Columns count            : {len(training_df.columns)}")
    pd.set_option("display.max_columns", 8)
    pd.set_option("display.width", 120)
    print("\nPreview (first 3 rows):")
    print(training_df.head(3))

if __name__ == "__main__":
    main()
