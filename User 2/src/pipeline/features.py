import argparse
import os
import sys
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Feature Engineering")
    parser.add_argument("--input", default="data/processed/preprocessed_data.csv", help="Input preprocessed CSV path")
    parser.add_argument("--output", default="data/processed/featured_data.csv", help="Output featured CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file '{args.input}' not found!")
        sys.exit(1)

    print(f"[*] Starting Feature Engineering from: {args.input}...")
    df = pd.read_csv(args.input)
    target_col = "target_label"

    # Identify numeric candidate features (excluding target)
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col]
    
    # Filter out constant / zero-variance columns (especially relevant for digits dataset)
    valid_cols = [c for c in numeric_cols if df[c].std() > 1e-6]
    if len(valid_cols) < 2:
        valid_cols = numeric_cols[:2]

    col_a = valid_cols[0]
    col_b = valid_cols[1] if len(valid_cols) > 1 else valid_cols[0]
    print(f"  Selected interaction columns: '{col_a}' and '{col_b}'")

    # Engineer 3 new features
    # 1. Product interaction
    df["feat_interaction_prod"] = df[col_a] * df[col_b]

    # 2. Ratio interaction (with epsilon to prevent division by zero)
    df["feat_interaction_ratio"] = df[col_a] / (df[col_b].abs() + 1e-5)

    # 3. Quantile binning feature
    try:
        df["feat_quantile_bin"] = pd.qcut(df[col_a], q=4, labels=False, duplicates="drop")
        df["feat_quantile_bin"] = df["feat_quantile_bin"].fillna(0).astype(int)
    except Exception:
        df["feat_quantile_bin"] = 0

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.output, index=False)

    print(f"[SUCCESS] Feature engineering complete.")
    print(f"  Output path         : {args.output}")
    print(f"  Total columns       : {len(df.columns)}")
    print(f"  Engineered features : ['feat_interaction_prod', 'feat_interaction_ratio', 'feat_quantile_bin']")

if __name__ == "__main__":
    main()
