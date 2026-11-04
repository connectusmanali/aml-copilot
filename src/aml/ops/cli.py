"""aml-ops drift                      PSI of features and scores: training period vs latest week
aml-ops audit-verify               verify the audit hash chain in Postgres
aml-ops replay --case CASE_ID      re-run a case from recorded inputs and diff the case file
aml-ops shadow --challenger DIR    compare a challenger model with the current one
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from aml.config import get_settings


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aml-ops")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drift")
    d.add_argument("--days", type=float, default=7.0, help="size of the current window")
    sub.add_parser("audit-verify")
    r = sub.add_parser("replay")
    r.add_argument("--case", required=True)
    sh = sub.add_parser("shadow")
    sh.add_argument("--challenger", type=Path, required=True)
    args = p.parse_args(argv)
    s = get_settings()

    if args.cmd == "audit-verify":
        from aml.cases.audit import verify
        from aml.cases.store import PostgresStore

        chain = verify(PostgresStore(s.postgres_dsn).audit())
        print(json.dumps(chain.__dict__))
        sys.exit(0 if chain.ok else 1)

    if args.cmd == "replay":
        from aml.cases.store import PostgresStore
        from aml.gateway.gateway import build_gateway
        from aml.ops.replay import load_replay, replay_case
        from aml.workflow.worker import case_data

        row = PostgresStore(s.postgres_dsn).latest_case_file(args.case)
        if row is None:
            sys.exit(f"no case file for {args.case}")
        ok, changed = replay_case(
            row["payload"],
            case_data(),
            load_replay(s.postgres_dsn, args.case),
            build_gateway().provider,
        )
        print("reproduced exactly" if ok else f"differs in: {changed}")
        sys.exit(0 if ok else 1)

    from aml.detection.model import ModelBundle
    from aml.features.pipeline import TRAINING_FILE

    frame = pd.read_parquet(s.features_dir / TRAINING_FILE)
    champion = ModelBundle.load(s.models_dir)
    if args.cmd == "drift":
        from aml.ops.drift import drift_report

        start, end = champion.meta["periods"]["train"]
        ref = frame[(frame["ts"] >= start) & (frame["ts"] <= end)]
        cur = frame[frame["ts"] > frame["ts"].max() - pd.Timedelta(days=args.days)]
        drift = drift_report(
            ref,
            cur,
            champion.features,
            champion.predict(ref),
            champion.predict(cur),
            champion.threshold,
        )
        print(drift.render())
        sys.exit(2 if drift.worst == "act" else 0)

    from aml.ops.shadow import shadow

    challenger = ModelBundle.load(args.challenger.parent, args.challenger.name)
    _, rep = shadow(frame, champion, challenger, frame["is_laundering"].astype(int))
    print(json.dumps({**rep.__dict__, "promotable": rep.promotable()}, indent=2))


if __name__ == "__main__":
    main()
