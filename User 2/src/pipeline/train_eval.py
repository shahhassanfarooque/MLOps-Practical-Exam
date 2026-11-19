import argparse
import json
import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Stage 5: Model Training and Evaluation")
    parser.add_argument("--input", default="data/processed/featured_data.csv", help="Input featured CSV path")
    parser.add_argument("--model-out", default="models/pipeline_model.joblib", help="Output model path")
    parser.add_argument("--metrics-out", default="metrics/eval_metrics.json", help="Output metrics JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file '{args.input}' not found!", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Starting Model Training and Evaluation using: {args.input}...")
    df = pd.read_csv(args.input)
    target_col = "target_label"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    print(f"  Dataset Shape : {df.shape}")
    print(f"  Features      : {len(feature_cols)}")
    print(f"  Classes       : {sorted(y.unique().tolist())}")

    # 80/20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train set     : {len(X_train)} samples")
    print(f"  Test set      : {len(X_test)} samples")

    # Train RandomForestClassifier
    print("[*] Training RandomForestClassifier(n_estimators=100, random_state=42)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions & metrics
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": len(feature_cols)
    }

    # Save model artifact
    model_dir = os.path.dirname(args.model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, args.model_out)
    print(f"[OK] Model serialized to: {args.model_out}")

    # Save evaluation metrics
    metrics_dir = os.path.dirname(args.metrics_out)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved to: {args.metrics_out}")

    print("\n================ EVALUATION METRICS ================")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  Precision Macro : {metrics['precision_macro']:.4f}")
    print(f"  Recall Macro    : {metrics['recall_macro']:.4f}")
    print(f"  F1 Score Macro  : {metrics['f1_macro']:.4f}")
    print("====================================================")
    print("[SUCCESS] Stage 5 Model training and evaluation finished.")

if __name__ == "__main__":
    main()
