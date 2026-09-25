"""Run a local Qwen Jev-style decision endpoint on a single selected GPU."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from .qwen_logits import engine_from_environment
from .schemas import DecisionRequest, DecisionResponse


app = FastAPI(title="Jev4Rec local Qwen decision service", version="0.1.0")


@lru_cache(maxsize=1)
def get_engine():
    return engine_from_environment()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend": "qwen-logits",
        "device": os.environ.get("JEV_DEVICE", "cuda:0"),
        "model": os.environ.get("JEV_QWEN_MODEL", "Qwen/Qwen3-14B"),
    }


@app.post("/v1/systemone", response_model=DecisionResponse)
def system_one(request: DecisionRequest) -> DecisionResponse:
    try:
        return get_engine().decide(request)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
