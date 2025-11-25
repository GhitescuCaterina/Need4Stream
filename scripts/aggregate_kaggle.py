
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"          
OUT  = PROC / "monthly_panel_template.csv"  

EU_COUNTRIES = [
    "Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina",
    "Bulgaria","Croatia","Cyprus","Czech Republic","Denmark","Estonia","Finland","France","Georgia","Germany",
    "Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Kosovo","Latvia","Liechtenstein","Lithuania",
    "Luxembourg","Malta","Moldova, Rep.","Monaco","Montenegro","Netherlands","North Macedonia","Norway","Poland",
    "Portugal","Romania","Russian Federation","San Marino","Serbia","Slovakia","Slovenia","Spain","Sweden",
    "Switzerland","Turkey","Ukraine","United Kingdom","Vatican City"
]

def info(msg: str):
    print(f"[aggregate] {msg}")

def to_monthly_mean(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    g = df.groupby(df[date_col].dt.to_period("M")).mean(numeric_only=True)
    g.index = g.index.to_timestamp()
    g = g.rename_axis("date").reset_index()
    return g

def annual_pairs_to_monthly(df: pd.DataFrame, year_col: str, value_col: str, label: str) -> pd.DataFrame:
    t = df[[year_col, value_col]].dropna().copy()
    t[year_col] = pd.to_numeric(t[year_col], errors="coerce")
    t[value_col] = pd.to_numeric(t[value_col], errors="coerce")
    t = t.dropna(subset=[year_col, value_col])
    t["date"] = pd.to_datetime(t[year_col].astype(int).astype(str) + "-01-01")
    t = t[["date", value_col]].set_index("date").resample("MS").ffill().reset_index()
    return t.rename(columns={value_col: label})

def broadcast_by_country(df_no_country: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """Cartesian product (countries x df_no_country)."""
    df_no_country = df_no_country.copy()
    df_no_country["key"] = 1
    cc = pd.DataFrame({"country": countries})
    cc["key"] = 1
    out = cc.merge(df_no_country, on="key", how="inner").drop(columns=["key"])
    return out

def first_csv(name_keywords: list[str]) -> Path | None:
    """Return first CSV in PROC whose filename contains all keywords (case-insensitive)."""
    for p in PROC.rglob("*.csv"):
        name = p.name.lower()
        if all(k.lower() in name for k in name_keywords):
            return p
    return None

def build_cpi_all_countries() -> pd.DataFrame:
    cpi_file = None
    for p in PROC.rglob("*.csv"):
        name = p.name.lower()
        if any(k in name for k in ["cpi", "consumer_price", "hicp", "inflation"]):
            cpi_file = p
            break
    if not cpi_file:
        raise SystemExit("CPI: no CSV found in data/processed/ (need filename containing 'cpi'/'hicp'/'inflation').")

    info(f"CPI source: {cpi_file.name}")
    cpi_raw = pd.read_csv(cpi_file)
    cpi_raw.columns = [str(c).strip() for c in cpi_raw.columns]
    if cpi_raw.empty or len(cpi_raw.columns) < 2:
        raise SystemExit(f"CPI: unexpected format in {cpi_file.name} (empty or <2 columns).")

    year_col = cpi_raw.columns[0]
    present = [c for c in EU_COUNTRIES if c in cpi_raw.columns]
    if not present:
        preview = cpi_raw.columns[1:11].tolist()
        raise SystemExit(f"CPI: no EU country columns found in {cpi_file.name}. Example columns: {preview}")

    df = cpi_raw[[year_col] + present].copy()
    long = df.melt(id_vars=[year_col], var_name="country", value_name="CPI")
    long[year_col] = pd.to_numeric(long[year_col], errors="coerce")
    long["CPI"] = pd.to_numeric(long["CPI"], errors="coerce")
    long = long.dropna(subset=[year_col, "CPI"])
    long["date"] = pd.to_datetime(long[year_col].astype(int).astype(str) + "-01-01")
    long = long[["country","date","CPI"]].set_index("date")

    pieces = []
    for c, g in long.groupby("country"):
        gm = g[["CPI"]].resample("MS").ffill()
        gm["country"] = c
        pieces.append(gm.reset_index())
    cpi_all = pd.concat(pieces, ignore_index=True).sort_values(["country","date"])
    info(f"CPI OK: {cpi_all['country'].nunique()} countries, {len(cpi_all)} rows")
    return cpi_all 

def build_pricebasket_monthly() -> pd.DataFrame:
    import pandas as pd
    import re

    price_file = PROC / "streaming_service.csv"
    if not price_file.exists():
        raise SystemExit(f"Prices: {price_file.name} nu există în {PROC}")

    info(f"Prices source: {price_file.name}")

    def load_no_header():
        df0 = pd.read_csv(price_file, header=None, names=["service","period","price"])
        first_period = str(df0.iloc[0,1])
        if re.fullmatch(r"[A-Za-z]{3}-\d{4}", first_period):
            return df0
        return None

    df = load_no_header()
    if df is None:
        df = pd.read_csv(price_file)
        cols = [c.strip().lower() for c in df.columns]
        mapping = {}
        for i,c in enumerate(cols):
            if c in ("service","platform","provider"):
                mapping[df.columns[i]] = "service"
            elif c in ("period","month","date","time"):
                mapping[df.columns[i]] = "period"
            elif c in ("price","price_usd","price_eur","cost"):
                mapping[df.columns[i]] = "price"
        df = df.rename(columns=mapping)
        if not set(["service","period","price"]).issubset(df.columns):
            df = df.iloc[:, :3].copy()
            df.columns = ["service","period","price"]

    df["service"] = df["service"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["period"].astype(str).str.strip(), format="%b-%Y", errors="coerce")
    mask_bad = df["date"].isna()
    if mask_bad.any():
        df.loc[mask_bad, "date"] = pd.to_datetime(df.loc[mask_bad, "period"], errors="coerce")

    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(",", "."), errors="coerce")

    df = df.dropna(subset=["date","price"])

    prices_m = (
        df.groupby(df["date"].dt.to_period("M"))["price"]
          .sum()
          .to_timestamp()
          .reset_index()
          .rename(columns={"price":"PriceBasket"})
          .sort_values("date")
    )

    info(f"PriceBasket monthly: {len(prices_m)} rows (from {price_file.name})")
    return prices_m


def build_catalog_indices_monthly() -> pd.DataFrame:
    import pandas as pd

    cat_file = PROC / "MoviesOnStreamingPlatforms.csv"  
    if not cat_file.exists():
        raise SystemExit(f"Catalog: {cat_file.name} nu există în {PROC}")

    info(f"Catalog source: {cat_file.name}")
    cat = pd.read_csv(cat_file)
    cat.columns = [c.strip() for c in cat.columns]

    if "Year" not in cat.columns:
        raise SystemExit("Catalog: lipsă coloană 'Year' în MoviesOnStreamingPlatforms.csv")

    platforms = []
    for c in cat.columns:
        cl = c.lower()
        if cl in ("netflix","hulu","prime video","prime_video","disney+","disney_plus"):
            platforms.append(c)
    if not platforms:
        raise SystemExit("Catalog: nu detectez coloane platformă ('Netflix','Hulu','Prime Video','Disney+').")

    def binarize(x):
        try:
            return 1 if int(float(x)) == 1 else 0
        except Exception:
            return 0

    dfp = cat.copy()
    for c in platforms:
        dfp[c] = dfp[c].apply(binarize)

    by_year = {}
    for y, grp in dfp.groupby("Year"):
        counts = {c: int(grp[c].sum()) for c in platforms}
        total = sum(counts.values()) or 1
        shares = {k: v/total for k, v in counts.items()}
        hhi = sum(s**2 for s in shares.values())
        exclusives = (grp[platforms].sum(axis=1) == 1).sum()
        exclusivity_share = exclusives / max(total,1)
        by_year[y] = {"HHI_titles": hhi, "ExclusivityShare": exclusivity_share}

    hhi_df = pd.DataFrame.from_dict(by_year, orient="index").reset_index().rename(columns={"index":"year"})
    hhi_df["date"] = pd.to_datetime(hhi_df["year"].astype(int).astype(str) + "-01-01")
    hhi_m = hhi_df.drop(columns=["year"]).set_index("date").resample("MS").ffill().reset_index()

    info(f"Catalog (HHI/Exclusivity) monthly: {len(hhi_m)} rows")
    return hhi_m


def main():
    info("Start aggregation for EU multi-country panel...")
    if not PROC.exists():
        raise SystemExit(f"Processed folder not found: {PROC}")

    cpi_all = build_cpi_all_countries()          

    prices_m = build_pricebasket_monthly()       
    countries = sorted(cpi_all["country"].unique().tolist())
    prices_bc = broadcast_by_country(prices_m, countries)   

    hhi_m = build_catalog_indices_monthly()     
    hhi_bc = broadcast_by_country(hhi_m, countries)

    panel = (
        cpi_all.merge(prices_bc, on=["country","date"], how="outer")
               .merge(hhi_bc,    on=["country","date"], how="outer")
               .sort_values(["country","date"])
               .reset_index(drop=True)
    )

    required_cols = [
        "GoogleTrends_torrent","GoogleTrends_watch_free","Lumen_DMCA_count",
        "PiracyIndex","Income","PriceIncome","PlatformCount",
        "event_price_hike","event_dns_block","event_major_release",
        "PiracyIndex_lag1","PiracyIndex_lag2","PiracyIndex_lag3",
        "PriceBasket_lag1","PriceBasket_lag2","PriceBasket_lag3",
        "rolling_mean_3","month_sin","month_cos","FragmentationIndex"
    ]
    for col in required_cols:
        if col not in panel.columns:
            panel[col] = pd.NA

    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT, index=False)
    info(f"Wrote: {OUT}")
    info("Open the CSV, check values, fill events (0/1), and add Income when you have it.")

if __name__ == "__main__":
    main()
