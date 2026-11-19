import argparse
import os
import sys
import numpy as np
import pandas as pd

class DataValidationError(Exception):
    """Custom exception raised when pipeline data validation assertions fail."""
    pass

def validate_dataframe(df: pd.DataFrame) -> None:
    print("[*] Running Stage 4 Data Validation checks...")

    # Check 1: Target column presence
    target_col = "target_label"
    if target_col not in df.columns:
        raise DataValidationError(f"Target column '{target_col}' is missing from the dataset!")

    # Check 2: Engineered features presence
    expected_engineered = ["feat_interaction_prod", "feat_interaction_ratio", "feat_quantile_bin"]
    for feat in expected_engineered:
        if feat not in df.columns:
            raise DataValidationError(f"Expected engineered feature '{feat}' is missing!")

    # Check 3: Zero unexpected null values across the entire dataset
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        raise DataValidationError(f"Dataset contains unexpected null values: {columns_with_nulls.to_dict()}")

    # Check 4: Valid target classes
    unique_labels = df[target_col].dropna().unique()
    if len(unique_labels) < 2:
        raise DataValidationError(f"Dataset must have at least 2 target classes, found: {unique_labels}")

    # Check 5: Check for infinite or extreme NaN corrupted values in numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if np.isinf(numeric_df.values).any():
        raise DataValidationError("Dataset contains infinite (inf / -inf) corrupted numeric values!")

    print("[OK] Check 1: Target column present.")
    print("[OK] Check 2: Engineered features verified.")
    print("[OK] Check 3: Zero null values detected.")
    print(f"[OK] Check 4: Valid classification labels ({len(unique_labels)} classes: {sorted(list(unique_labels))}).")
    print("[OK] Check 5: No infinite values or numerical corruption found.")

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Data Validation and Quality Gate")
    parser.add_argument("--input", default="data/processed/featured_data.csv", help="Input featured CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file '{args.input}' not found!", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.input)
        validate_dataframe(df)
        print(f"[SUCCESS] All Stage 4 data validation checks passed for {len(df)} rows.")
    except DataValidationError as e:
        print(f"\n[VALIDATION FAILURE] DataValidationError: {e}", file=sys.stderr)
        print("[HARD STOP] Halting ML pipeline execution due to data quality gate failure.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
