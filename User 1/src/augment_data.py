import os
import sys
import numpy as np
import pandas as pd

# Ensure dataset_loader can be imported from root of part-1-dvc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_loader import DATASET_NAME

def main():
    target_path = os.path.join("data", "raw", "dataset_v1.csv")
    if not os.path.exists(target_path):
        print(f"[ERROR] Target file '{target_path}' does not exist! Run 'python src/generate_data.py' first.")
        sys.exit(1)
        
    print(f"[*] Reading Version 1 dataset from: {target_path}...")
    df = pd.read_csv(target_path)
    initial_rows = len(df)
    target_col = "target_label"
    feature_cols = [c for c in df.columns if c != target_col]
    
    print(f"[*] Original dataset shape: {df.shape}")
    print("[*] Sampling 20 random rows (random_state=42) for data augmentation...")
    sample_df = df.sample(n=20, random_state=42).copy()
    
    print("[*] Adding Gaussian noise N(0, 0.05) to numeric feature columns...")
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.05, size=(20, len(feature_cols)))
    
    # Add noise to numeric features
    for idx, col in enumerate(feature_cols):
        sample_df[col] = sample_df[col].astype(float) + noise[:, idx]
        
    print("[*] Concatenating augmented rows to produce Version 2...")
    df_v2 = pd.concat([df, sample_df], ignore_index=True)
    df_v2.to_csv(target_path, index=False)
    
    print("[SUCCESS] Version 2 dataset generated and saved!")
    print(f"  Target File    : {target_path}")
    print(f"  Initial Rows   : {initial_rows}")
    print(f"  Augmented Rows : {len(sample_df)}")
    print(f"  Updated Rows   : {len(df_v2)} (Expected {initial_rows + 20})")
    print(f"  Columns        : {len(df_v2.columns)}")

if __name__ == "__main__":
    main()
