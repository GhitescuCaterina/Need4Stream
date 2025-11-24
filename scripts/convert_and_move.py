from pathlib import Path
import pandas as pd
import re, time, shutil

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "data" / "raw"
DST = PROJECT / "data" / "processed"

DST.mkdir(parents=True, exist_ok=True)

def safe_name(name: str) -> str:
    name = name.strip()
    name = name.replace(".adj..", ".")   
    name = re.sub(r"\s+", "_", name)    
    name = re.sub(r"[^A-Za-z0-9_.-]", "", name) 
    name = name.replace("__", "_")
    return name

def unique_path(dst_dir: Path, filename: str) -> Path:
    p = dst_dir / filename
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix or ".csv"
    ts = time.strftime("%Y%m%d_%H%M%S")
    return dst_dir / f"{stem}__{ts}{suf}"

def move_csv_file(src_csv: Path):
    out = unique_path(DST, safe_name(src_csv.name))
    shutil.copy2(src_csv, out)
    print(f"MOVED CSV : {src_csv.relative_to(PROJECT)} -> {out.relative_to(PROJECT)}")

def convert_excel(xlsx_path: Path):
    try:
        xls = pd.ExcelFile(xlsx_path)
    except Exception as e:
        print(f"SKIP  XLSX: {xlsx_path.relative_to(PROJECT)} ({e})")
        return
    base = safe_name(xlsx_path.stem)
    wrote_any = False
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
            if df.empty:
                continue
            sheet_tag = safe_name(sheet) or "Sheet"
            out_name = f"{base}__{sheet_tag}.csv"
            out = unique_path(DST, out_name)
            df.to_csv(out, index=False)
            print(f"CONVERTED: {xlsx_path.relative_to(PROJECT)} [{sheet}] -> {out.relative_to(PROJECT)}")
            wrote_any = True
        except Exception as e:
            print(f"SKIP SHEET: {xlsx_path.name} [{sheet}] ({e})")
    if not wrote_any:
        print(f"NO DATA  : {xlsx_path.relative_to(PROJECT)} (all sheets empty)")

def main():
    if not SRC.exists():
        raise SystemExit(f"Source not found: {SRC}")
    for x in SRC.rglob("*"):
        if x.is_file() and x.suffix.lower() in (".xlsx", ".xls"):
            convert_excel(x)
    for c in SRC.rglob("*.csv"):
        try:
            if DST in c.parents:
                continue
            move_csv_file(c)
        except Exception as e:
            print(f"SKIP CSV : {c.relative_to(PROJECT)} ({e})")
    print("\nDONE. Check:", DST)

if __name__ == "__main__":
    main()
