"""Agent command line.

aml-agent investigate --txn <txn_id>   # case file as JSON (uses data/processed + alerts)
aml-agent index-policies               # embed policy guidance into pgvector
"""

from __future__ import annotations

import argparse
import sys

from aml.agent.data import FrameData
from aml.agent.graph import AgentConfig, InvestigationAgent
from aml.agent.policy import PgVectorPolicyIndex
from aml.config import get_settings
from aml.gateway.gateway import build_gateway


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aml-agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    ip = sub.add_parser("investigate", help="investigate one alert and print the case file")
    ip.add_argument("--txn", required=True)
    sub.add_parser("index-policies", help="embed policy guidance into ref.policy_chunks")
    args = p.parse_args(argv)

    s = get_settings()
    gateway = build_gateway()
    if args.cmd == "index-policies":
        n = PgVectorPolicyIndex(s.postgres_dsn, gateway.provider).build()
        print(f"indexed {n} policy chunks with {gateway.provider.model}")
        return
    alerts = s.features_dir / "alerts.parquet"
    if not alerts.exists():
        sys.exit("no alerts: run `make score` first")
    agent = InvestigationAgent(
        FrameData.from_dirs(s.processed_dir, s.features_dir),
        gateway,
        config=AgentConfig(s.agent_max_tool_calls, s.agent_max_seconds, s.case_token_budget),
    )
    print(agent.investigate(args.txn).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
