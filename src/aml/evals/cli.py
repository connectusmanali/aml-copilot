"""aml-eval run [--size eval|smoke]    full evaluation; non-zero exit if any gate fails
aml-eval judge-template               drafts + empty grade columns for human graders
aml-eval judge-calibrate --grades F   judge vs human agreement (needs an LLM provider)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aml.evals import harness, judge
from aml.gateway.gateway import build_gateway
from aml.report.draft import draft_report

OUT = Path("evals/results")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aml-eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--size", choices=["eval", "smoke"], default="eval")
    r.add_argument("--out", type=Path, default=OUT)
    t = sub.add_parser("judge-template")
    t.add_argument("--out", type=Path, default=Path("evals/human_grades.csv"))
    c = sub.add_parser("judge-calibrate")
    c.add_argument("--grades", type=Path, default=Path("evals/human_grades.csv"))
    args = p.parse_args(argv)

    gateway = build_gateway()
    mode = gateway.provider.name
    if args.cmd == "run":
        res = harness.run(args.size, gateway if mode != "offline" else None)
        path = harness.save(res, args.out, mode)
        print(path.read_text())
        sys.exit(0 if res.ok else 1)
    if args.cmd == "judge-template":
        w = harness.build_world("smoke")
        runs = harness.run_agent(w, harness.golden_set(w, per_kind=4))
        drafts = {c.case_id: draft_report(cf).text for c, cf, _ in runs if c.kind != "starved"}
        print(f"wrote {judge.write_template(drafts, args.out)}: fill in the grade columns")
        return
    if mode == "offline":
        sys.exit("judge calibration needs a real LLM: set AML_LLM_PROVIDER and AML_LLM_API_KEY")
    print(json.dumps(judge.calibrate(gateway, args.grades), indent=2))


if __name__ == "__main__":
    main()
