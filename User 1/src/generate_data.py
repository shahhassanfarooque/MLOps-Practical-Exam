import os
import sys

# Ensure dataset_loader can be imported from root of part-1-dvc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_loader import DATASET_NAME, get_dataset

def main():
    print(f"[*] Loading active dataset: '{DATASET_NAME}'...")
    df, feature_cols, target_col = get_dataset()
    
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "dataset_v1.csv")
    df.to_csv(output_path, index=False)
    
    print("[SUCCESS] Version 1 dataset generated successfully!")
    print(f"  Destination : {output_path}")
    print(f"  Rows        : {len(df)}")
    print(f"  Columns     : {len(df.columns)} ({len(feature_cols)} features, 1 target: '{target_col}')")
    print(f"  Classes     : {sorted(df[target_col].unique().tolist())}")

if __name__ == "__main__":
    main()
