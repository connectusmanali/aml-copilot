"""aml-api serve [--demo N] [--temporal]

--demo N     in-memory store seeded with the top N alerts, run through agent + drafting
             (no Postgres or Temporal needed; uses data/processed and data/features)
--temporal   send decisions to the case workflows instead of writing them directly
"""

from __future__ import annotations

import argparse

import pandas as pd
import uvicorn

from aml.api.app import create_app, temporal_decider
from aml.config import get_settings


def demo_service(n: int):  # type: ignore[no-untyped-def]
    from aml.agent.data import FrameData
    from aml.cases.service import CaseService
    from aml.cases.store import MemoryStore
    from aml.workflow.worker import alert_payload

    s = get_settings()
    data = FrameData.from_dirs(s.processed_dir, s.features_dir)
    svc = CaseService(MemoryStore(), data, sla_hours=s.case_sla_hours)
    alerts = pd.read_parquet(s.features_dir / "alerts.parquet")
    top = alerts[alerts["priority"] != "none"].sort_values("score", ascending=False)
    for row in pd.concat([top.head(n), top[top["priority"] == "low"].head(3)]).to_dict("records"):
        a = alert_payload(row)
        cid = svc.open(a)
        if a["priority"] == "high":
            svc.investigate(cid)
            svc.draft(cid)
        else:
            svc.light_review(cid)
    return svc


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aml-api")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("serve")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--demo", type=int, default=0, metavar="N")
    sp.add_argument("--temporal", action="store_true")
    args = p.parse_args(argv)
    s = get_settings()
    if args.demo:
        service = demo_service(args.demo)
    else:
        from aml.workflow.worker import build_service

        service = build_service()
    decider = temporal_decider(s.temporal_host) if args.temporal else None
    uvicorn.run(create_app(service, decider), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
