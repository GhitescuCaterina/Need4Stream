# notebooks/build_index.py
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
INP  = PROC / "monthly_panel_template.csv"
OUT  = PROC / "monthly_panel.csv"

def zscore(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(skipna=True), s.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / sd

def build_fragmentation(hhi):
    # higher fragmentation when HHI is low
    return 1.0 - (hhi - np.nanmin(hhi)) / (np.nanmax(hhi) - np.nanmin(hhi) + 1e-9)

def make_proxy_piracy(df):
    """
    If PiracyIndex is missing/NaN, build a proxy:
      + z(PriceBasket)    (more services/prices -> more incentive to pirate)
      + z(Fragmentation)  (more fragmentation -> more incentive)
      + 0.3*z(CPI)        (inflation pressure)
      - 0.3*z(Income)     (if Income exists)
    If GoogleTrends_torrent exists, blend: 0.6*proxy + 0.4*z(trends).
    """
    parts = []
    if "PriceBasket" in df.columns:
        parts.append(zscore(df["PriceBasket"]))
    if "FragmentationIndex" in df.columns:
        parts.append(zscore(df["FragmentationIndex"]))
    if "CPI" in df.columns:
        parts.append(0.3 * zscore(df["CPI"]))
    if "Income" in df.columns and df["Income"].notna().any():
        parts.append(-0.3 * zscore(df["Income"]))

    proxy = sum(parts) if parts else pd.Series(np.zeros(len(df)), index=df.index)

    if "GoogleTrends_torrent" in df.columns and df["GoogleTrends_torrent"].notna().any():
        proxy = 0.6 * proxy + 0.4 * zscore(df["GoogleTrends_torrent"])

    return (proxy - np.nanmin(proxy)) / (np.nanmax(proxy) - np.nanmin(proxy) + 1e-9)

def add_time_feats(df):
    df = df.sort_values("date").copy()
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12.0)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12.0)
    return df

def add_lags_roll(df, col, lags=(1,2,3), roll=3):
    for L in lags:
        df[f"{col}_lag{L}"] = df[col].shift(L)
    df["rolling_mean_3"] = df[col].rolling(roll, min_periods=1).mean()
    return df

def main():
    if not INP.exists():
        raise SystemExit(f"Missing: {INP}")
    panel = pd.read_csv(INP, parse_dates=["date"])
    if "country" not in panel.columns:
        # Single-country fallback: add a dummy
        panel["country"] = "Romania"

    # Minimal cleaning
    for c in ["CPI","PriceBasket","HHI_titles","ExclusivityShare","FragmentationIndex","Income"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    # If FragmentationIndex not present, derive from HHI_titles
    if "FragmentationIndex" not in panel.columns or panel["FragmentationIndex"].isna().all():
        if "HHI_titles" in panel.columns and panel["HHI_titles"].notna().any():
            # 1 - normalized HHI
            g = panel.groupby("country", group_keys=False)
            panel["FragmentationIndex"] = g["HHI_titles"].apply(build_fragmentation)
        else:
            panel["FragmentationIndex"] = np.nan

    out_rows = []
    for country, g in panel.groupby("country"):
        g = g.sort_values("date").copy()

        # Trim to real price coverage
        if "PriceBasket" in g.columns and g["PriceBasket"].notna().any():
            start = g.loc[g["PriceBasket"].notna(), "date"].min()
            end   = g.loc[g["PriceBasket"].notna(), "date"].max()
            g = g[(g["date"] >= start) & (g["date"] <= end)].copy()
        else:
            continue  # no price coverage → skip country

        # Forward-only smoothing (no backfill to the past)
        for c in ["CPI","PriceBasket","HHI_titles","ExclusivityShare","FragmentationIndex","Income"]:
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce")
                g[c] = g[c].interpolate(limit_direction="forward").ffill()

        # Time feats
        g = add_time_feats(g)

        # Fragmentation from HHI if missing
        if ("FragmentationIndex" not in g.columns) or g["FragmentationIndex"].isna().all():
            if "HHI_titles" in g.columns and g["HHI_titles"].notna().any():
                g["FragmentationIndex"] = (1 - (g["HHI_titles"] - g["HHI_titles"].min()) /
                                            (g["HHI_titles"].max() - g["HHI_titles"].min() + 1e-9))

        # Target (proxy) if missing/empty
        if "PiracyIndex" not in g.columns or g["PiracyIndex"].isna().all():
            g["PiracyIndex"] = make_proxy_piracy(g)

        # Drop if constant target after build
        if g["PiracyIndex"].nunique(dropna=True) < 2:
            continue

        # Lags
        g = add_lags_roll(g, "PiracyIndex", lags=(1,2,3), roll=3)
        if "PriceBasket" in g.columns:
            for L in (1,2,3):
                g[f"PriceBasket_lag{L}"] = g["PriceBasket"].shift(L)

        # Events default to 0
        for ev in ["event_price_hike","event_dns_block","event_major_release"]:
            if ev not in g.columns:
                g[ev] = 0

        out_rows.append(g)

    final = (pd.concat(out_rows, ignore_index=True)
            .dropna(subset=["PiracyIndex"])  # ensure target present
            .sort_values(["country","date"]))
    final.to_csv(OUT, index=False)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
