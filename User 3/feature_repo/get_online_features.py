import os
import sys
import pandas as pd
from feast import FeatureStore

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    store = FeatureStore(repo_path=repo_dir)

    entity_rows = [
        {"sample_id": 0},
        {"sample_id": 1},
        {"sample_id": 2},
    ]

    print("[*] Querying SQLite Online Store for low-latency inference features...")
    print(f"  Target sample_ids: [0, 1, 2]")

    feature_service = store.get_feature_service("exam_feature_service")
    response = store.get_online_features(
        features=feature_service,
        entity_rows=entity_rows,
    )

    df_online = response.to_df()
    print("\n[SUCCESS] Retrieved Online Features from SQLite:")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(df_online)
    print(f"\nDataFrame shape: {df_online.shape}")

if __name__ == "__main__":
    main()
