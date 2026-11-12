"""Step 8: investigator API, roles, and the full human-in-the-loop path over HTTP."""

import pytest
from fastapi.testclient import TestClient

from aml.api.app import create_app
from aml.cases.service import CaseService
from aml.cases.store import MemoryStore
from tests.test_agent import world  # noqa: F401
from tests.test_workflow import alert_row

INV, SEN, AUD = {"X-API-Key": "k-inv"}, {"X-API-Key": "k-sen"}, {"X-API-Key": "k-aud"}
KEYS = "k-inv=alice:investigator,k-sen=bob:senior,k-aud=carol:auditor"


@pytest.fixture()
def setup(world: dict) -> tuple[TestClient, str]:  # noqa: F811
    svc = CaseService(MemoryStore(), world["data"])
    cid = svc.open(alert_row(world, "Cash"))
    svc.investigate(cid)
    svc.draft(cid)
    return TestClient(create_app(svc, api_keys=KEYS)), cid


def test_auth_and_roles(setup: tuple[TestClient, str]) -> None:
    client, cid = setup
    assert client.get("/cases").status_code == 401
    assert client.get("/cases", headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/cases", headers=AUD).status_code == 200
    r = client.post(
        f"/cases/{cid}/decision", json={"decision": "close", "reason": "x"}, headers=AUD
    )
    assert r.status_code == 403
    assert client.get("/audit/verify", headers=INV).status_code == 403


def test_full_review_path(setup: tuple[TestClient, str]) -> None:
    client, cid = setup
    detail = client.get(f"/cases/{cid}", headers=INV).json()
    assert detail["case"]["status"] == "in_review" and detail["case_file"]["evidence"]
    draft = detail["reports"][0]["body"]

    r = client.post(
        f"/cases/{cid}/decision", json={"decision": "file_str", "reason": "x"}, headers=INV
    )
    assert r.status_code == 409 and "final report" in r.json()["detail"]

    bad = client.post(
        f"/cases/{cid}/report", json={"body": draft + "\nUnsupported claim.\n"}, headers=INV
    ).json()
    assert bad["validation"]["ok"] is False
    assert client.post(f"/cases/{cid}/report/approve", headers=INV).status_code == 409

    edited = draft.replace(
        "## Action",
        f"## Action\nThe investigator reviewed the file [alert:{detail['case']['txn_id']}].",
    )
    ok = client.post(f"/cases/{cid}/report", json={"body": edited}, headers=INV).json()
    assert ok["validation"]["ok"] is True
    assert client.post(f"/cases/{cid}/report/approve", headers=INV).status_code == 200
    r = client.post(
        f"/cases/{cid}/decision",
        json={"decision": "file_str", "reason": "structuring confirmed"},
        headers=INV,
    )
    assert r.json()["status"] == "filed"
    assert (
        client.post(
            f"/cases/{cid}/feedback", json={"agrees_with_score": True}, headers=INV
        ).status_code
        == 200
    )

    d = client.get(f"/cases/{cid}", headers=AUD).json()["decision"]
    assert 0 < d["edit_distance"] < 0.2
    actions = [e["action"] for e in client.get(f"/cases/{cid}/audit", headers=AUD).json()]
    assert actions[0] == "opened" and "report_final" in actions and "feedback" in actions
    assert client.get("/audit/verify", headers=AUD).json()["ok"] is True
