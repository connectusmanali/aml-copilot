"""Scoring API. Stateless: the model bundle is loaded once at startup.

POST /score   {"rows": [ {training_frame columns...}, ... ]}
GET  /model   version, threshold, feature list, test metrics
GET  /health
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aml.config import get_settings
from aml.detection.model import ModelBundle
from aml.detection.scoring import score_frame


class ScoreRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1_000)


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    app = FastAPI(title="AML risk scorer", version="1.0")
    state: dict[str, ModelBundle] = {}

    def model() -> ModelBundle:
        if "bundle" not in state:
            state["bundle"] = bundle or ModelBundle.load(get_settings().models_dir)
        return state["bundle"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model().version}

    @app.get("/model")
    def model_info() -> dict[str, Any]:
        b = model()
        return {
            "version": b.version,
            "threshold": b.threshold,
            "features": b.features,
            "meta": b.meta,
        }

    @app.post("/score")
    def score(req: ScoreRequest) -> dict[str, Any]:
        frame = pd.DataFrame(req.rows)
        try:
            scored = score_frame(frame, model())
        except KeyError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        records = scored.assign(ts=scored["ts"].astype(str)).to_dict(orient="records")
        for r in records:
            r["reasons"] = json.loads(r["reasons"]) if r.get("reasons") else []
        return {"model_version": model().version, "results": records}

    return app
