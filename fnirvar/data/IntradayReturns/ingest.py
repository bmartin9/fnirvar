""" 
Script to proess the .7z Intraday returns data and store as a parquet file.
The output is a data lake with the following structure:
data_parquet/asset=<TICKER>/ym=<YYYY-MM>/<YYYY-MM-DD>.parquet

Only level-1 quotes (bid_1 / ask_1) are stored.
"""

#!/usr/bin/env python

from __future__ import annotations
import io, pathlib, re, sys, os
from concurrent.futures import ProcessPoolExecutor
import polars as pl
from py7zr import SevenZipFile
import tempfile
import logging

RAW_DIR      = pathlib.Path("../../../data/IntradayReturns/raw/data_by_stocks/")
PARQUET_DIR  = pathlib.Path("data_parquet")
USD_SCALE    = 10_000
EXPECTED_MINUTES = {391, 210}

# ───────────────  logging for row-count anomalies  ────────────────
logging.basicConfig(
    filename="ingest_warnings.log",   # file will be created/append-to
    filemode="a",
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

# --------------------------------------------------------------------------- #
#  Regex helpers
# --------------------------------------------------------------------------- #
#  Inside a daily CSV name      :  <TICKER>_YYYY-MM-DD_......
DAY_RE   = re.compile(r"^([A-Z]+)_(\d{4}-\d{2}-\d{2})_")
#  Whole archive file name      :  .....__<TICKER>_YYYY-... .7z
TICKER_RE = re.compile(r"([A-Z]+)_\d{4}-\d{2}-\d{2}")

# --------------------------------------------------------------------------- #
def load_day_csv(buf: bytes, csv_name: str) -> tuple[str, pl.DataFrame]:
    """Parse one day-CSV.  Returns (date_str, df)."""
    m = DAY_RE.match(csv_name)
    if not m:
        raise ValueError(f"CSV filename not recognised: {csv_name}")
    ticker, date_str = m.groups()

    df = (
    pl.read_csv(
        io.BytesIO(buf),
        has_header=True,
        schema_overrides={
            "time": pl.Utf8, "bid_1": pl.Int32, "ask_1": pl.Int32,
        },
        null_values=["", "NaN", "-9999999999","9999999999"]
    )
    .with_columns(
        # ───── correct timestamp construction ───────────────
        (
            (pl.lit(date_str + " ") + pl.col("time"))        # e.g. "2007-06-27 09:30:00"
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            .dt.replace_time_zone("UTC")
            .alias("ts")
        ),

        # ───── midpoint price ───────────────────────────────
        ((pl.col("bid_1") + pl.col("ask_1")) * 0.5 / USD_SCALE)
        .cast(pl.Float32)
        .alias("mid_px"),
    )
    .select(["ts", "mid_px"])
    .sort("ts")
    .with_columns(pl.col("mid_px").fill_null(strategy="forward"))
)


    if df.height not in EXPECTED_MINUTES:
        logging.warning(
            "Ticker %s  Date %s  had %d rows (expected 391 or 210)",
            m.group(1),           # ticker symbol captured by DAY_RE
            date_str,
            df.height,
        )

    return date_str, df


def write_day(df: pl.DataFrame, asset: str, date_str: str) -> None:
    ym = date_str[:7]
    out = PARQUET_DIR / asset / ym 
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"{date_str}.parquet", compression="snappy")


# --------------------------------------------------------------------------- #
# ────────────────────────────────────────────────────────────────────────────
#  ARCHIVE-LEVEL WORKER  (temp-folder method, works on every py7zr version)
# ────────────────────────────────────────────────────────────────────────────
def process_archive(archive_path: pathlib.Path) -> None:
    m = TICKER_RE.search(archive_path.stem)
    asset = m.group(1) if m else archive_path.stem
    print(f"[{asset}]  start", flush=True)

    try:
        with SevenZipFile(archive_path, mode="r") as z:
            member_names = z.getnames()

            with tempfile.TemporaryDirectory() as tmpdir:
                for csv_name in member_names:
                    # ➊  extract just this file
                    z.extract(path=tmpdir, targets=[csv_name])

                    csv_path = pathlib.Path(tmpdir) / csv_name
                    with open(csv_path, "rb") as fh:
                        date_str, df_day = load_day_csv(fh.read(), csv_name)
                        write_day(df_day, asset, date_str)

                    # ➋  immediately free the handle & inode
                    os.remove(csv_path)

    except Exception as exc:
        print(f"[{asset}]  ERROR: {exc}", flush=True)
        raise

    print(f"[{asset}]  done", flush=True)


# --------------------------------------------------------------------------- #
def archives_to_process(cli_args) -> list[pathlib.Path]:
    return [pathlib.Path(p) for p in cli_args] if cli_args \
           else sorted(RAW_DIR.glob("*.7z"))


if __name__ == "__main__":
    paths = archives_to_process(sys.argv[1:])
    if not paths:
        sys.exit("No .7z archives to ingest.")
    with ProcessPoolExecutor() as pool:
        pool.map(process_archive, paths)
