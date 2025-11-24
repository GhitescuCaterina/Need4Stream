import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_FEATURES = [
    "PriceBasket","PriceIncome","CPI","FragmentationIndex",
    "PiracyIndex_lag1","PiracyIndex_lag2","PiracyIndex_lag3",
    "PriceBasket_lag1","PriceBasket_lag2","PriceBasket_lag3",
    "rolling_mean_3","event_price_hike","event_dns_block","month_sin","month_cos"
]

def _forward_fill_by_country(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if "country" in df.columns:
        # forward-fill within each country
        filled = (
            df.sort_values(["country","date"])
              .groupby("country")[cols]
              .apply(lambda g: g.ffill())
        )
        # groupby returns a MultiIndex; align back
        filled.index = filled.index.droplevel(0)
        return filled
    else:
        return df.sort_values("date")[cols].ffill()

def run_tree_baseline(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "PiracyIndex" not in df.columns:
        raise SystemExit("CSV has no 'PiracyIndex' column. Build it in notebooks/build_index.py first.")

    # Feature list (drop PriceIncome if Income missing)
    features = BASE_FEATURES.copy()
    if ("Income" not in df.columns) or df["Income"].isna().all():
        if "PriceIncome" in features:
            features.remove("PriceIncome")
    features = [f for f in features if f in df.columns]

    # Cast numeric
    for c in set(features + ["PiracyIndex"]):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Forward-fill features within country
    df_ff = df.copy()
    if features:
        filled = _forward_fill_by_country(df_ff, features)
        df_ff.loc[:, features] = filled.values

    # Keep rows with target present
    df_ff = df_ff.dropna(subset=["PiracyIndex"]).copy()

    # Filter usable features: enough data and some variance
    usable_feats = []
    for f in features:
        col = df_ff[f]
        non_null = col.notna().sum()
        if non_null < 2:
            continue  # too sparse to compute variance
        var = np.nanvar(col.values)
        non_null_ratio = non_null / len(df_ff) if len(df_ff) else 0.0
        if non_null_ratio >= 0.6 and var > 1e-12:
            usable_feats.append(f)

    if not usable_feats:
        raise SystemExit("No usable features after cleaning (too many NaNs or constant). Check monthly_panel.csv.")

    # Final drop of rows with NaNs in chosen features
    df_ff = df_ff.dropna(subset=usable_feats).copy()

    # Build X, y
    X = df_ff[usable_feats].astype(float)
    y = df_ff["PiracyIndex"].astype(float)

    # Target variance check
    if y.nunique(dropna=True) < 2 or float(np.nanstd(y)) == 0.0:
        diag = {
            "rows_after_clean": int(len(df_ff)),
            "usable_features": usable_feats,
            "y_min": float(np.nanmin(y)) if len(y) else None,
            "y_max": float(np.nanmax(y)) if len(y) else None,
            "y_unique": int(y.nunique(dropna=True))
        }
        raise SystemExit(f"Target has no variance after preprocessing. Diagnostic: {diag}")

    # TimeSeriesSplit: ensure enough splits for the data
    # Require at least (n_splits+1) folds; fall back gracefully
    max_splits = 5
    n_splits = min(max_splits, max(2, len(X) // 60))  # ~60 pts per fold heuristic
    if n_splits < 2:
        n_splits = 2
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid = {
        "max_depth": [3,5,7,9],
        "min_samples_leaf": [5,20,50],
        "min_samples_split": [2,10]
    }

    model = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        grid,
        cv=tscv,
        scoring="neg_mean_absolute_error"
    )
    model.fit(X, y)

    best = model.best_estimator_
    pred_in = best.predict(X)

    mae = mean_absolute_error(y, pred_in)
    # Older sklearn: no 'squared' param — compute RMSE manually
    rmse = float(np.sqrt(mean_squared_error(y, pred_in)))

    imp = pd.Series(best.feature_importances_, index=usable_feats).sort_values(ascending=False)

    return {
        "best_params": model.best_params_,
        "MAE_in": mae,
        "RMSE_in": rmse,
        "importances": imp.to_dict(),
        "used_features": usable_feats,
        "rows_trained": int(len(X)),
        "cv_splits": n_splits
    }

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to monthly_panel.csv")
    args = ap.parse_args()
    out = run_tree_baseline(args.csv)
    print(json.dumps(out, indent=2))
