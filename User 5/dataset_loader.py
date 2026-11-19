import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, load_digits

# CHANGE THIS ONE STRING AT RUNTIME: "wine", "breast_cancer", or "digits"
DATASET_NAME = "wine"

def get_dataset(name: str = DATASET_NAME):
    if name == "wine":
        bunch = load_wine(as_frame=True)
        df = bunch.frame.copy()
        target_col = "target_label"
        df[target_col] = df["target"].map(dict(enumerate(bunch.target_names)))
        df.drop(columns=["target"], inplace=True)
    elif name == "digits":
        bunch = load_digits(as_frame=True)
        df = bunch.frame.copy()
        target_col = "target_label"
        df[target_col] = df["target"].astype(str)
        df.drop(columns=["target"], inplace=True)
    elif name == "breast_cancer":
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=17)
            df = ds.data.features.copy()
            target_col = "target_label"
            raw_y = ds.data.targets.iloc[:, 0]
            df[target_col] = raw_y.map({"M": "malignant", "B": "benign"}).fillna(raw_y.astype(str))
        except Exception:
            bunch = load_breast_cancer(as_frame=True)
            df = bunch.frame.copy()
            target_col = "target_label"
            df[target_col] = df["target"].map(dict(enumerate(bunch.target_names)))
            df.drop(columns=["target"], inplace=True)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: wine, breast_cancer, digits")

    df.rename(
        columns={c: c.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") for c in df.columns},
        inplace=True,
    )
    feature_cols = [c for c in df.columns if c != target_col]
    return df, feature_cols, target_col
