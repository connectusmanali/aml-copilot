"""Data command line.

aml-data prepare --source synthetic            # no download needed
aml-data prepare --source ibm --size small     # needs data/raw/HI-Small_Trans.csv
aml-data load                                  # into Postgres (needs `make up`)
aml-data quality --from parquet|db             # data quality report; exit code 1 on failure
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from aml.config import get_settings
from aml.data import ibm, synthetic
from aml.data.load import load_all
from aml.data.profiles import build_profiles
from aml.data.quality import QualityReport, check_transactions
from aml.data.validate import validate_customers, validate_transactions
from aml.data.watchlist import build_watchlist

IBM_FILE = "HI-Small_Trans.csv"


def prepare(
    source: str, size: ibm.Size, raw_file: Path | None, out_dir: Path | None = None
) -> None:
    s = get_settings()
    t0 = time.perf_counter()

    if source == "synthetic":
        n_acc, n_norm, n_pat = synthetic.SIZES[size]
        raw = synthetic.generate(n_acc, n_norm, n_pat, seed=s.random_seed)
    else:
        path = raw_file or s.raw_dir / IBM_FILE
        if not path.exists():
            sys.exit(f"{path} not found. Run `make download` or pass --file.")
        # Read a little more than needed: invalid rows are rejected during validation.
        n = ibm.SIZES[size]
        raw = ibm.read_raw(path, nrows=None if n is None else int(n * 1.05))

    txns = ibm.take_window(ibm.normalize(raw), size if source == "ibm" else "full")
    valid, rejected = validate_transactions(txns)
    customers, accounts = build_profiles(valid, seed=s.random_seed)
    customers, bad_customers = validate_customers(customers)
    if len(bad_customers):  # generated profiles must always pass: a failure here is a bug
        raise RuntimeError(f"profile generator produced invalid customers:\n{bad_customers}")

    report = check_transactions(valid, accounts=accounts, rejected_count=len(rejected))
    print(report.render())
    if not report.ok:  # quality gate: nothing is written when a critical check fails
        sys.exit(1)

    out = out_dir or s.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    valid.to_parquet(out / "transactions.parquet", index=False)
    customers.to_parquet(out / "customers.parquet", index=False)
    accounts.to_parquet(out / "accounts.parquet", index=False)
    build_watchlist(valid, accounts, seed=s.random_seed).to_parquet(
        out / "watchlist.parquet", index=False
    )
    rejected.to_parquet(out / "rejected.parquet", index=False)
    (out / "_meta.json").write_text(json.dumps({"source": source, "size": size}))

    print(f"source={source} size={size}")
    print(f"  transactions: {len(valid):,} valid, {len(rejected):,} rejected")
    print(
        f"  laundering:   {int(valid['is_laundering'].sum()):,} "
        f"({valid['is_laundering'].mean():.3%})"
    )
    print(f"  customers:    {len(customers):,}")
    print(f"  time window:  {valid['ts'].min()} -> {valid['ts'].max()}")
    print(f"  wrote {out}/ in {time.perf_counter() - t0:.1f}s")


def quality(source: str) -> QualityReport:
    s = get_settings()
    if source == "db":
        from aml.db.read import ACCOUNTS_SQL, TRANSACTIONS_SQL, fetch_count, fetch_frame

        txns = fetch_frame(s.postgres_dsn, TRANSACTIONS_SQL)
        accounts = fetch_frame(s.postgres_dsn, ACCOUNTS_SQL)
        rejected = fetch_count(s.postgres_dsn, "SELECT count(*) FROM raw.rejected_records")
    else:
        d = s.processed_dir
        txns = pd.read_parquet(d / "transactions.parquet")
        accounts = pd.read_parquet(d / "accounts.parquet")
        rejected = len(pd.read_parquet(d / "rejected.parquet"))
    report = check_transactions(txns, accounts=accounts, rejected_count=rejected)
    print(report.render())
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aml-data")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="normalize, validate, build profiles -> Parquet")
    pp.add_argument("--source", choices=["synthetic", "ibm"], default="synthetic")
    pp.add_argument("--size", choices=list(ibm.SIZES), default="small")
    pp.add_argument("--file", type=Path, default=None)

    sub.add_parser("load", help="load Parquet into Postgres (idempotent)")

    qp = sub.add_parser("quality", help="data quality report (exit code 1 on failure)")
    qp.add_argument("--from", dest="source", choices=["parquet", "db"], default="parquet")

    args = p.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args.source, args.size, args.file)
    elif args.cmd == "quality":
        if not quality(args.source).ok:
            sys.exit(1)
    else:
        s = get_settings()
        load_all(s.postgres_dsn, s.processed_dir)


if __name__ == "__main__":
    main()
