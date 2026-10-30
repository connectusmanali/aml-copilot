"""The agent's read-only tools. Each returns Evidence with a typed, citable id.

Budgets are enforced HERE, at the tool boundary, not by asking the model to be careful: the
runner counts calls and elapsed time and raises before a tool runs past the limit.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from aml.agent.casefile import Evidence
from aml.agent.data import CaseData
from aml.agent.policy import PolicyIndex

LOOKBACK = timedelta(days=14)
HORIZON = timedelta(days=2)  # investigations happen after the alert; allow 2 days of follow-up
MAX_NEIGHBOURS = 25

# Instructions hidden in customer-controlled text (memos, names, occupations). Matching text is
# still shown to the analyst as DATA, and the case gets an open question.
INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above) instructions|disregard .{0,30}instructions|"
    r"system prompt|you are now|close (this|the) case|mark .{0,20}(clean|not suspicious))",
    re.I,
)


class ToolBudgetExceeded(RuntimeError):
    pass


def fmt_txn(t: dict[str, Any]) -> str:
    usd = t.get("amount_usd")
    amount = f"USD {usd:,.2f}" if isinstance(usd, int | float) else f"{t['amount_paid']} "
    return f"{t['ts'][:16]} · {t['from_account']} → {t['to_account']} · {amount} · " + str(
        t["payment_format"]
    )


def txn_evidence(t: dict[str, Any]) -> Evidence:
    return Evidence(id=f"txn:{t['txn_id']}", kind="transaction", summary=fmt_txn(t), data=t)


def find_injection(*texts: object) -> list[str]:
    return [str(t) for t in texts if isinstance(t, str) and INJECTION.search(t)]


class ToolRunner:
    def __init__(
        self,
        data: CaseData,
        policy: PolicyIndex,
        max_calls: int,
        max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.data, self.policy = data, policy
        self.max_calls, self.max_seconds, self.clock = max_calls, max_seconds, clock
        self.started = clock()
        self.calls: list[str] = []

    def _guard(self, name: str) -> None:
        if len(self.calls) >= self.max_calls:
            raise ToolBudgetExceeded(f"tool-call budget of {self.max_calls} reached before {name}")
        if self.clock() - self.started > self.max_seconds:
            raise ToolBudgetExceeded(
                f"time budget of {self.max_seconds:.0f}s reached before {name}"
            )
        self.calls.append(name)

    # ---- tools ----
    def get_alert(self, txn_id: str) -> list[Evidence]:
        self._guard("get_alert")
        alert = self.data.alert(txn_id)
        if alert is None:
            return []
        rules = alert.get("rules") or "none"
        summary = f"Alert on {txn_id}: model score {alert['score']:.3f}, rules fired: {rules}"
        return [Evidence(id=f"alert:{txn_id}", kind="alert", summary=summary, data=alert)]

    def get_customer_profile(self, account_id: str) -> list[Evidence]:
        self._guard("get_customer_profile")
        p = self.data.account(account_id)
        if p is None:
            return []
        summary = (
            f"{account_id}: {p['segment']} ({p['occupation']}), {p['country']}, "
            f"risk {p['risk_rating']}, declared USD {p['expected_monthly_volume']:,.0f}/month, "
            f"customer since {str(p['onboarded_at'])[:10]}"
        )
        return [Evidence(id=f"acct:{account_id}", kind="profile", summary=summary, data=p)]

    def get_transactions(self, account_id: str, around: datetime) -> list[Evidence]:
        self._guard("get_transactions")
        rows = self.data.transactions(account_id, around - LOOKBACK, around + HORIZON)
        return [txn_evidence(t) for t in rows]

    def screen_watchlist(self, account_ids: list[str]) -> list[Evidence]:
        self._guard("screen_watchlist")
        return [
            Evidence(
                id=f"wl:{h['entry_id']}",
                kind="watchlist",
                summary=f"{h['account_id']} on {h['list']} list: {h['reason']}",
                data=h,
            )
            for h in self.data.watchlist(sorted(set(account_ids)))
        ]

    def explore_network(self, account_id: str, around: datetime) -> list[Evidence]:
        """Counterparties and three-account loops (S -> B -> C -> S, in time order)."""
        self._guard("explore_network")
        since, until = around - LOOKBACK, around + HORIZON
        own = self.data.transactions(account_id, since, until, limit=500)
        outs = [t for t in own if t["from_account"] == account_id]
        ins = [t for t in own if t["to_account"] == account_id]
        neighbours = list(dict.fromkeys(t["to_account"] for t in outs))[:MAX_NEIGHBOURS]

        cycles: list[list[dict[str, Any]]] = []
        for b in neighbours:
            b_out = [
                t
                for t in self.data.transactions(b, since, until, limit=200)
                if t["from_account"] == b and t["to_account"] != account_id
            ]
            for t2 in b_out:
                c = t2["to_account"]
                for t3 in self.data.transactions(c, since, until, limit=200):
                    if t3["from_account"] != c or t3["to_account"] != account_id:
                        continue
                    for t1 in outs:
                        if t1["to_account"] == b and t1["ts"] <= t2["ts"] <= t3["ts"]:
                            cycles.append([t1, t2, t3])
            if len(cycles) >= 5:
                break

        evidence = [txn_evidence(t) for cyc in cycles for t in cyc]
        summary = (
            f"{account_id}: paid {len({t['to_account'] for t in outs})} distinct accounts and was "
            f"paid by {len({t['from_account'] for t in ins})} in the window; "
            f"{len(cycles)} loop(s) back to the account found"
        )
        data = {
            "out_counterparties": len({t["to_account"] for t in outs}),
            "in_counterparties": len({t["from_account"] for t in ins}),
            "cycles": [[t["txn_id"] for t in cyc] for cyc in cycles],
        }
        return [
            Evidence(id=f"net:{account_id}", kind="network", summary=summary, data=data),
            *evidence,
        ]

    def search_policy(self, query: str, k: int = 2) -> list[Evidence]:
        self._guard("search_policy")
        return [
            Evidence(
                id=f"pol:{c.chunk_id}",
                kind="policy",
                summary=f"{c.doc} / {c.heading}",
                data={"text": c.body, "score": round(score, 3)},
            )
            for c, score in self.policy.search(query, k)
        ]
