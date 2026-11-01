"""Data quality checks that stop the pipeline when the data is wrong.

Validation (validate.py) judges one record at a time. Quality checks judge the dataset as a
whole: duplicates, nulls, label rates, unknown currencies, missing profiles. A batch of
individually valid rows can still be a broken dataset.

Severity:
  critical -> the pipeline stops (prepare exits with an error; nothing is written)
  warning  -> reported and kept; a person should look
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

from aml.data.fx import to_usd, unknown_currencies

REQUIRED = [
    "txn_id",
    "ts",
    "from_account",
    "to_account",
    "amount_paid",
    "payment_currency",
    "amount_received",
    "receiving_currency",
    "payment_format",
    "is_laundering",
]
Severity = Literal["critical", "warning"]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    severity: Severity
    observed: str
    expected: str


@dataclass(frozen=True)
class QualityReport:
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(c.passed or c.severity == "warning" for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def render(self) -> str:
        lines = [f"{'check':<28} {'result':<9} {'observed':<26} expected"]
        for c in self.checks:
            result = "ok" if c.passed else ("FAIL" if c.severity == "critical" else "warn")
            lines.append(f"{c.name:<28} {result:<9} {c.observed:<26} {c.expected}")
        lines.append("PASSED" if self.ok else "FAILED: critical checks did not pass")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}


def check_transactions(
    txns: pd.DataFrame,
    *,
    accounts: pd.DataFrame | None = None,
    rejected_count: int = 0,
    now: datetime | None = None,
    max_rejected_rate: float = 0.01,
    laundering_rate_band: tuple[float, float] = (0.0001, 0.05),
) -> QualityReport:
    now = now or datetime.now()
    n = len(txns)
    checks: list[Check] = []

    def add(name: str, passed: bool, severity: Severity, observed: object, expected: str) -> None:
        checks.append(Check(name, bool(passed), severity, str(observed), expected))

    add("row_count", n > 0, "critical", n, "> 0")
    if n == 0:
        return QualityReport(checks)

    dupes = int(txns["txn_id"].duplicated().sum())
    add("duplicate_txn_ids", dupes == 0, "critical", dupes, "0")

    nulls = {c: int(txns[c].isna().sum()) for c in REQUIRED if c in txns}
    missing_cols = [c for c in REQUIRED if c not in txns]
    null_total = sum(nulls.values())
    observed = f"{null_total} ({', '.join(k for k, v in nulls.items() if v) or 'none'})"
    add("required_nulls", null_total == 0 and not missing_cols, "critical", observed, "0")

    total_in = n + rejected_count
    rej_rate = rejected_count / total_in
    add(
        "rejected_rate",
        rej_rate <= max_rejected_rate,
        "warning",
        f"{rej_rate:.3%}",
        f"<= {max_rejected_rate:.1%}",
    )

    unknown = unknown_currencies(pd.concat([txns["payment_currency"], txns["receiving_currency"]]))
    add(
        "unknown_currencies",
        not unknown,
        "warning",
        ", ".join(unknown) or "none",
        "none (add to fx.USD_PER_UNIT)",
    )

    rate = float(txns["is_laundering"].astype(bool).mean())
    lo, hi = laundering_rate_band
    add("laundering_label_rate", lo <= rate <= hi, "warning", f"{rate:.3%}", f"{lo:.2%}-{hi:.0%}")

    future = int((pd.to_datetime(txns["ts"]) > pd.Timestamp(now)).sum())
    add("future_timestamps", future == 0, "critical", future, "0")

    span = pd.to_datetime(txns["ts"]).max() - pd.to_datetime(txns["ts"]).min()
    add("time_span", span >= pd.Timedelta(days=1), "warning", str(span), ">= 1 day")

    usd = to_usd(txns["amount_paid"], txns["payment_currency"])
    biggest = float(usd.max()) if usd.notna().any() else 0.0
    add("max_amount_usd", biggest <= 1e9, "warning", f"{biggest:,.0f}", "<= 1,000,000,000")

    if accounts is not None:
        used = set(txns["from_account"]) | set(txns["to_account"])
        orphans = len(used - set(accounts["account_id"]))
        add("accounts_without_profile", orphans == 0, "critical", orphans, "0")

    return QualityReport(checks)
