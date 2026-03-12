from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from calibration_utils import FEATURE_COLUMNS


def train_screen_mapper(
    csv_path="calibration_samples.csv",
    out_x_path="model_x.pkl",
    out_y_path="model_y.pkl",
    alpha=5.0,
):
    csv_path = Path(csv_path)
    out_x_path = Path(out_x_path)
    out_y_path = Path(out_y_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = FEATURE_COLUMNS + ["target_x", "target_y"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in calibration CSV: {missing}")

    if len(df) < 15:
        raise ValueError(
            f"Not enough calibration rows: {len(df)}. "
            f"Collect at least 15 to 27+ rows for a stable mapper."
        )

    X = df[FEATURE_COLUMNS].values
    y_x = df["target_x"].values
    y_y = df["target_y"].values

    model_x = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    model_y = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    model_x.fit(X, y_x)
    model_y.fit(X, y_y)

    joblib.dump(model_x, out_x_path)
    joblib.dump(model_y, out_y_path)

    print(f"Saved x model to {out_x_path}")
    print(f"Saved y model to {out_y_path}")
    print(f"Trained on {len(df)} rows")

    if "phase" in df.columns:
        print("\nRows by phase:")
        print(df["phase"].value_counts())

    print("\nFeature columns used:")
    for col in FEATURE_COLUMNS:
        print(f"- {col}")


if __name__ == "__main__":
    train_screen_mapper()