#!/usr/bin/env python
# snapshot_diagnostics.py
#
# Scans snapshot folders and monthly parquet bars to report
# * IPO and delisting months
# * missing 30-minute bars per (ticker, month)

from __future__ import annotations
import pathlib, json, datetime as dt, itertools
import polars as pl
from collections import defaultdict, Counter
from dateutil.relativedelta import relativedelta

ROOT = pathlib.Path("~/phd/projects/factors/fnirvar/fnirvar/data/IntradayReturns").expanduser()
SNAP = ROOT / "snapshots"
BARS = ROOT / "data_bars" / "30m"
OUT  = pathlib.Path(__file__).resolve().parent / "diagnostics_out"
OUT.mkdir(exist_ok=True)

# ---------- pass 1: first/last snapshot month per ticker -------------------
first_seen: dict[str, str] = {}
last_seen : dict[str, str] = {}

for sdir in sorted(SNAP.iterdir()):
    if not sdir.is_dir():
        continue
    month = sdir.name                 # YYYY-MM-DD
    tickers = pl.read_csv(sdir / "coverage.csv")["ticker"]
    for tkr in tickers:
        first_seen.setdefault(tkr, month)
        last_seen[tkr] = month        # keeps updating

ipo_rows = []
for tkr in first_seen:
    ipo_rows.append((tkr, first_seen[tkr], last_seen[tkr]))
pl.DataFrame(ipo_rows, schema=["ticker","first_month","last_month"]) \
  .write_csv(OUT / "ipo_delist.csv")

# quick yearly counts
ipo_years  = Counter(m[:4] for m in first_seen.values())
del_years  = Counter(m[:4] for m in last_seen.values())
print("IPO per year :", dict(ipo_years))
print("Delist per year :", dict(del_years))

# ---------- pass 2: missing-bar matrix ------------------------------------
missing = []   # rows: (month, ticker, missing_count)

# def expected_rows(month: str) -> int:
#     y,m,_ = month.split("-")
#     # count NYSE full trading days in that month
#     first = dt.date(int(y),int(m),1)
#     nextm = first.replace(day=28) + dt.timedelta(days=4)
#     lastd = nextm - dt.timedelta(days=nextm.day)
#     days  = sum(1 for d in range(lastd.day)
#                   if (first+dt.timedelta(d)).weekday()<5)  # Mon-Fri
#     return days * 14   # 14 regular-session bars

def expected_rows(month: str, ref_ticker: str = "SPY") -> int:
    """Count the actual rows present in a reference ticker for that month."""
    pf = BARS / ref_ticker / f"{month[:7]}.parquet"
    return pl.read_parquet(pf).shape[0] if pf.exists() else 0


for tkr, first_m, last_m in ipo_rows:
    start = dt.datetime.fromisoformat(first_m[:7] + "-01")
    stop  = dt.datetime.fromisoformat(last_m [:7] + "-01")

    ym = start
    while ym <= stop:
        ym_str = ym.strftime("%Y-%m")
        pf = BARS / tkr / f"{ym_str}.parquet"
        if pf.exists():
            nrows = pl.read_parquet(pf).shape[0]
            miss  = expected_rows(ym_str + "-01") - nrows
            if miss:
                missing.append((ym_str, tkr, miss))
        ym += relativedelta(months=+1)

pl.DataFrame(missing, schema=["month","ticker","missing_bars"]) \
  .write_parquet(OUT / "missing_bars.parquet")

print("Diagnostics written to", OUT)
