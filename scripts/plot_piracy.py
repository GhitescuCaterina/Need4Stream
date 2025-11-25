import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_eu(df, outdir: Path):
    eu = (df.groupby("date", as_index=False)["PiracyIndex"]
            .mean()
            .rename(columns={"PiracyIndex":"PiracyIndex_EU"}))
    eu["PiracyIndex_EU_roll6"] = eu["PiracyIndex_EU"].rolling(6, min_periods=1).mean()

    plt.figure(figsize=(10,5))
    plt.plot(eu["date"], eu["PiracyIndex_EU"], label="EU Avg PiracyIndex")
    plt.plot(eu["date"], eu["PiracyIndex_EU_roll6"], label="EU 6-mo rolling", linestyle="--")
    plt.title("Need4Stream • EU-wide PiracyIndex (monthly)")
    plt.xlabel("Date")
    plt.ylabel("PiracyIndex (0–1)")
    plt.legend()
    out = outdir / "piracy_eu_avg.png"
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    print(f"[saved] {out}")

def plot_country(df, country: str, outdir: Path):
    g = df[df["country"] == country].sort_values("date").copy()
    if g.empty:
        print(f"[warn] No rows for country={country}")
        return
    g["PiracyIndex_roll6"] = g["PiracyIndex"].rolling(6, min_periods=1).mean()

    plt.figure(figsize=(10,5))
    plt.plot(g["date"], g["PiracyIndex"], label=f"{country} PiracyIndex")
    plt.plot(g["date"], g["PiracyIndex_roll6"], label=f"{country} 6-mo rolling", linestyle="--")
    plt.title(f"Need4Stream • {country} PiracyIndex (monthly)")
    plt.xlabel("Date")
    plt.ylabel("PiracyIndex (0–1)")
    plt.legend()
    out = outdir / f"piracy_{country.lower().replace(' ','_')}.png"
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    print(f"[saved] {out}")

def main(csv_path: str, country: str):
    csv = Path(csv_path)
    outdir = csv.parent
    df = pd.read_csv(csv, parse_dates=["date"])
    for c in ("date","country","PiracyIndex"):
        if c not in df.columns:
            raise SystemExit(f"Missing '{c}' in {csv}")

    plot_eu(df, outdir)
    plot_country(df, country, outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/monthly_panel.csv")
    ap.add_argument("--country", default="Romania")
    args = ap.parse_args()
    main(args.csv, args.country)
