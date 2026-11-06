"""Investigator UI (Streamlit). Talks only to the API, never to the database.

uv run --extra ui streamlit run src/aml/ui/app.py
(API at AML_API_URL, default http://localhost:8000; key in the sidebar)
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API = os.environ.get("AML_API_URL", "http://localhost:8000")

st.set_page_config(page_title="AML Investigation Copilot", layout="wide")
key = st.sidebar.text_input("API key", value="dev-investigator", type="password")
client = httpx.Client(base_url=API, headers={"X-API-Key": key}, timeout=30)


def get(path: str, **params: Any) -> Any:
    r = client.get(path, params=params)
    if r.status_code >= 400:
        st.error(f"{r.status_code}: {r.json().get('detail')}")
        st.stop()
    return r.json()


def post(path: str, body: dict[str, Any] | None = None) -> tuple[bool, Any]:
    r = client.post(path, json=body)
    return r.status_code < 400, r.json()


me = get("/me")
st.sidebar.markdown(f"Signed in as **{me['id']}** ({me['role']})")
statuses = st.sidebar.multiselect(
    "Status",
    ["in_review", "needs_human", "escalated", "filed", "closed"],
    default=["in_review", "needs_human", "escalated"],
)
queue = get("/cases", status=",".join(statuses), limit=200)

st.title("Case queue")
if not queue:
    st.info("No cases in this view.")
    st.stop()
st.dataframe(
    [
        {
            "case": c["case_id"][:20],
            "account": c["account_id"],
            "score": round(c["score"], 3),
            "priority": c["priority"],
            "status": c["status"],
            "SLA due": str(c["sla_due_at"])[:16],
        }
        for c in queue
    ],
    width="stretch",
    hide_index=True,
)

case_id = st.selectbox("Open case", [c["case_id"] for c in queue])
d = get(f"/cases/{case_id}")
case, cf = d["case"], d["case_file"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Score", f"{case['score']:.3f}")
c2.metric("Status", case["status"])
c3.metric("Suggested", cf["suggested_disposition"] if cf else "light review")
c4.metric("Evidence items", len(cf["evidence"]) if cf else 0)

tab_findings, tab_report, tab_decide, tab_audit = st.tabs(
    ["Findings", "Report", "Decision", "Audit trail"]
)

with tab_findings:
    if cf is None:
        st.info("Low-priority alert: routed to light review without an agent investigation.")
    else:
        if cf["status"] == "needs_human":
            st.warning("The investigation is incomplete; see open questions.")
        st.subheader("Risk factors")
        for rf in cf["risk_factors"]:
            st.markdown(
                f"- **{rf['severity']}** · {rf['statement']}  `{', '.join(rf['evidence_ids'][:4])}`"
            )
        st.subheader("Open questions")
        for q in cf["open_questions"]:
            st.markdown(f"- {q}")
        with st.expander("All evidence"):
            for e in cf["evidence"]:
                st.markdown(f"`{e['id']}` {e['summary']}")

with tab_report:
    reports = d["reports"]
    latest = next((r for r in reversed(reports) if r["kind"] in {"edit", "draft"}), None)
    if latest is None:
        st.info("No draft for this case.")
    else:
        v = latest["validation"]
        (st.success if v.get("ok") else st.error)(
            f"Validator: {'pass' if v.get('ok') else 'FAIL'} · "
            f"citation rate {v.get('citation_rate', 0):.0%}"
        )
        for issue in v.get("issues", []):
            st.markdown(f"- :red[{issue['problem']}] — {issue['sentence'][:160]}")
        body = st.text_area(
            "Narrative (every sentence needs a [citation])", latest["body"], height=420
        )
        a, b = st.columns(2)
        if a.button("Save and validate"):
            ok, res = post(f"/cases/{case_id}/report", {"body": body})
            st.rerun()
        if b.button("Approve as final"):
            ok, res = post(f"/cases/{case_id}/report/approve")
            (st.success if ok else st.error)(res if ok else res["detail"])

with tab_decide:
    if d["decision"]:
        st.success(
            f"Decided: {d['decision']['decision']} by {d['decision']['decided_by']} — "
            f"{d['decision']['reason']}"
        )
    else:
        choice = st.radio(
            "Decision",
            ["close", "file_str"],
            horizontal=True,
            format_func=lambda x: {"close": "Close (not suspicious)", "file_str": "File STR"}[x],
        )
        reason = st.text_area("Reason (required, goes in the audit log)")
        agrees = st.checkbox("I agree with the risk score", value=True)
        if st.button("Submit decision", type="primary"):
            ok, res = post(f"/cases/{case_id}/decision", {"decision": choice, "reason": reason})
            post(f"/cases/{case_id}/feedback", {"agrees_with_score": agrees})
            (st.success if ok else st.error)(res.get("status") if ok else res["detail"])

with tab_audit:
    for e in get(f"/cases/{case_id}/audit"):
        st.markdown(
            f"`{e['at'][:19]}` **{e['actor']}** {e['action']} "
            f"<span style='color:gray'>#{e['row_hash'][:10]}</span>",
            unsafe_allow_html=True,
        )
