import argparse
import os
import sys
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Preprocess raw data (clean, deduplicate, impute)")
    parser.add_argument("--input", default="data/raw/raw_data.csv", help="Input raw CSV path")
    parser.add_argument("--output", default="data/processed/preprocessed_data.csv", help="Output preprocessed CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file '{args.input}' not found!")
        sys.exit(1)

    print(f"[*] Starting Preprocessing from: {args.input}...")
    df = pd.read_csv(args.input)
    initial_rows = len(df)
    target_col = "target_label"

    # 1. Drop duplicate rows
    df = df.drop_duplicates()
    dedup_rows = len(df)
    print(f"  Rows after deduplication: {dedup_rows} (dropped {initial_rows - dedup_rows} duplicates)")

    # 2. Identify feature columns (excluding target and timestamp)
    feature_cols = [c for c in df.columns if c not in (target_col, "collected_at")]

    # 3. Coerce feature columns to numeric and impute NaNs with column median
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)

    # 4. Drop rows with missing target label
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
        df[target_col] = df[target_col].astype(str)

    # 5. Remove 'collected_at' metadata column
    if "collected_at" in df.columns:
        df = df.drop(columns=["collected_at"])

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.output, index=False)

    print(f"[SUCCESS] Preprocessing completed.")
    print(f"  Output path      : {args.output}")
    print(f"  Final rows       : {len(df)}")
    print(f"  Feature columns  : {len(feature_cols)}")
    print(f"  Target column    : {target_col}")

if __name__ == "__main__":
    main()
