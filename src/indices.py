import numpy as np
import pandas as pd

def build_piracy_index(df: pd.DataFrame, components=None, weights=None, zscore=True):
    """
    Build PiracyIndex from selected components in df.
    components: list of column names (e.g., ["GoogleTrends_torrent","GoogleTrends_watch_free","Lumen_DMCA_count"])
    weights: list of floats same length as components; defaults to equal weights
    zscore: if True, z-score each component before averaging
    """
    if components is None:
        components = ["GoogleTrends_torrent","GoogleTrends_watch_free","Lumen_DMCA_count"]
    X = df[components].astype(float).copy()
    if zscore:
        X = (X - X.mean())/X.std(ddof=0)
    if weights is None:
        weights = np.ones(len(components))/len(components)
    df["PiracyIndex"] = (X * weights).sum(axis=1)
    return df

def build_fragmentation_index(df: pd.DataFrame):
    """
    Example fragmentation proxy combining ExclusivityShare and (1 - normalized HHI).
    Assumes columns: ExclusivityShare (0..1), HHI_titles (0..1)
    """
    frag = 0.5*df["ExclusivityShare"].astype(float) + 0.5*(1 - df["HHI_titles"].astype(float))
    df["FragmentationIndex"] = frag
    return df

def add_time_features(df: pd.DataFrame, date_col="date"):
    d = pd.to_datetime(df[date_col])
    month = d.dt.month
    df["month_sin"] = np.sin(2*np.pi*month/12)
    df["month_cos"] = np.cos(2*np.pi*month/12)
    return df

def add_lags(df: pd.DataFrame, col, lags=(1,2,3)):
    for L in lags:
        df[f"{col}_lag{L}"] = df[col].shift(L)
    return df

def add_rolling(df: pd.DataFrame, col, window=3):
    df[f"rolling_mean_{window}"] = df[col].rolling(window, min_periods=1).mean()
    return df
