"""Detection command line.

aml-detect rules                       # rule scenarios on the training frame
aml-detect train [--capacity 100]      # train + calibrate + evaluate + ablations + register
aml-detect score                       # score all transactions -> data/features/alerts.parquet
aml-detect serve [--port 8001]         # scoring API
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from aml.config import get_settings
from aml.detection.model import ModelBundle
from aml.detection.rules import evaluate, load_scenarios, rule_metrics
from aml.detection.scoring import score_frame
from aml.detection.train import ablations, log_to_mlflow, render_report, train_scorer
from aml.features.pipeline import TRAINING_FILE

ALERTS_FILE = "alerts.parquet"


def _frame() -> pd.DataFrame:
    path = get_settings().features_dir / TRAINING_FILE
    if not path.exists():
        sys.exit(f"{path} missing: run `make features` first")
    return pd.read_parquet(path)


def _mlflow_uri() -> str:
    """Configured server if reachable, else a local SQLite store (tracking + registry)."""
    s = get_settings()
    uri = s.mlflow_uri
    if uri.startswith("http"):
        import httpx

        try:
            httpx.get(f"{uri}/health", timeout=2.0).raise_for_status()
            return uri
        except Exception:
            pass
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(s.data_dir / 'mlflow.db').resolve()}"


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    p = argparse.ArgumentParser(prog="aml-detect")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rules", help="rule scenario metrics on the whole training frame")
    tp = sub.add_parser("train", help="train, calibrate, evaluate, register the scorer")
    tp.add_argument("--capacity", type=int, default=100, help="alerts investigators review/day")
    tp.add_argument("--no-ablation", action="store_true")
    sub.add_parser("score", help="score every transaction with the current model")
    sp = sub.add_parser("serve", help="run the scoring API")
    sp.add_argument("--port", type=int, default=8001)
    args = p.parse_args(argv)

    s = get_settings()
    if args.cmd == "rules":
        frame = _frame()
        days = (frame["ts"].max() - frame["ts"].min()) / pd.Timedelta(days=1)
        print(
            rule_metrics(evaluate(frame, load_scenarios()), frame["is_laundering"], days).to_string(
                index=False
            )
        )
    elif args.cmd == "train":
        frame = _frame()
        bundle, report = train_scorer(frame, alerts_per_day=args.capacity, seed=s.random_seed)
        abl = [] if args.no_ablation else ablations(frame, args.capacity, s.random_seed)
        out = bundle.save(s.models_dir)
        text = render_report(report, abl, bundle)
        (out / "evaluation.md").write_text(text)
        (out / "evaluation.json").write_text(
            json.dumps({"report": report, "ablations": abl}, indent=2, default=str)
        )
        run_id = log_to_mlflow(bundle, report, _mlflow_uri(), str(out))
        print(text)
        print(f"saved {out}" + (f" | mlflow run {run_id}" if run_id else " | mlflow not installed"))
    elif args.cmd == "score":
        bundle = ModelBundle.load(s.models_dir)
        scored = score_frame(_frame(), bundle)
        path = s.features_dir / ALERTS_FILE
        scored.to_parquet(path, index=False)
        counts = scored["priority"].value_counts().to_dict()
        print(f"scored {len(scored):,} with {bundle.version}: {counts} -> {path}")
    else:
        import uvicorn

        from aml.detection.service import create_app

        uvicorn.run(create_app(), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
