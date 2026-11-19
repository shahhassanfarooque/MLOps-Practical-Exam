import argparse
import os
import sys
from datetime import datetime, timezone

# Ensure dataset_loader can be imported from root of part-2-pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dataset_loader import DATASET_NAME, get_dataset

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Collect raw dataset with timestamp metadata")
    parser.add_argument("--output", default="data/raw/raw_data.csv", help="Path to save raw CSV output")
    args = parser.parse_args()

    print(f"[*] Starting Data Collection for dataset: '{DATASET_NAME}'...")
    df, feature_cols, target_col = get_dataset()

    # Append UTC ISO collection timestamp
    timestamp = datetime.now(timezone.utc).isoformat()
    df["collected_at"] = timestamp

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.output, index=False)

    print(f"[SUCCESS] Data collection complete.")
    print(f"  Output path    : {args.output}")
    print(f"  Rows collected : {len(df)}")
    print(f"  Features count : {len(feature_cols)}")
    print(f"  Target column  : {target_col}")
    print(f"  Timestamp UTC  : {timestamp}")

if __name__ == "__main__":
    main()
