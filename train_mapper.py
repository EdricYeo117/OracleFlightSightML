from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge

from calibration_utils import FEATURE_COLUMNS


def train_screen_mapper(
    csv_path="calibration_samples.csv",
    out_x_path="model_x.pkl",
    out_y_path="model_y.pkl",
):
    csv_path = Path(csv_path)
    out_x_path = Path(out_x_path)
    out_y_path = Path(out_y_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if len(df) < 5:
        raise ValueError("Not enough calibration rows. Collect more samples first.")

    X = df[FEATURE_COLUMNS].values
    y_x = df["target_x"].values
    y_y = df["target_y"].values

    model_x = Ridge(alpha=1.0)
    model_y = Ridge(alpha=1.0)

    model_x.fit(X, y_x)
    model_y.fit(X, y_y)

    joblib.dump(model_x, out_x_path)
    joblib.dump(model_y, out_y_path)

    print(f"Saved x model to {out_x_path}")
    print(f"Saved y model to {out_y_path}")


if __name__ == "__main__":
    train_screen_mapper()