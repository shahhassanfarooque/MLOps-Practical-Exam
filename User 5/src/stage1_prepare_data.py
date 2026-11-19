import argparse
import os
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# Allow importing dataset_loader from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_loader import DATASET_NAME, get_dataset

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Ingest, Clean, Engineer, and Validate Features")
    parser.add_argument(
        "--output",
        default="data/processed/validated_features.parquet",
        help="Path for validated output parquet file",
    )
    args = parser.parse_args()

    print(f"[*] [Stage 1] Ingesting dataset: '{DATASET_NAME}'...")
    df, raw_feature_cols, target_col = get_dataset()
    initial_rows = len(df)

    # 1. Deduplication
    df = df.drop_duplicates()
    print(f"  Rows after deduplication: {len(df)} (dropped {initial_rows - len(df)} duplicates)")

    # 2. Impute NaNs in numeric features with median
    for col in raw_feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[col].median()
        if pd.isna(med):
            med = 0.0
        df[col] = df[col].fillna(med).astype(float)

    # 3. Dynamic Feature Engineering: select top 2 non-constant features
    numeric_cols = [c for c in raw_feature_cols if df[c].std() > 1e-6]
    if len(numeric_cols) < 2:
        numeric_cols = raw_feature_cols[:2]

    col_a, col_b = numeric_cols[0], numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
    print(f"  Selected features for interaction: '{col_a}' and '{col_b}'")

    df["feat_interaction_prod"] = (df[col_a] * df[col_b]).astype(float)
    df["feat_interaction_ratio"] = (df[col_a] / (df[col_b].abs() + 1e-5)).astype(float)

    # 4. Strict Quality Gate & Schema Validation
    print("[*] Running Stage 1 Data Validation checks...")
    if df[target_col].isnull().any():
        print(f"[VALIDATION ERROR] Missing target values found in '{target_col}'!", file=sys.stderr)
        sys.exit(1)

    if df.isnull().sum().sum() > 0:
        print("[VALIDATION ERROR] Unexpected null values found in DataFrame!", file=sys.stderr)
        sys.exit(1)

    if np.isinf(df.select_dtypes(include=[np.number]).values).any():
        print("[VALIDATION ERROR] Infinite values found in feature matrix!", file=sys.stderr)
        sys.exit(1)

    num_classes = df[target_col].nunique()
    if num_classes < 2:
        print(f"[VALIDATION ERROR] Need at least 2 classes, found {num_classes}!", file=sys.stderr)
        sys.exit(1)

    print(f"  [OK] Quality gates passed: zero nulls, zero infs, {num_classes} valid classes.")

    # 5. Add Feast primary entity key and UTC timestamps
    df["sample_id"] = range(len(df))
    df["sample_id"] = df["sample_id"].astype("int64")

    now_utc = datetime.now(timezone.utc)
    df["event_timestamp"] = pd.to_datetime(now_utc)
    df["created_timestamp"] = pd.to_datetime(now_utc)

    # 6. Save Parquet
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_parquet(args.output, index=False)
    print(f"[SUCCESS] Stage 1 complete. Saved: {args.output} ({len(df)} rows, {len(df.columns)} columns)")

if __name__ == "__main__":
    main()
