"""Strict request and response models for the local decision endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str = Field(min_length=1)
    criteria: Dict[str, str] = Field(min_items=2, max_items=26)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str = Field(min_length=1)
    criteria: List[str] = Field(min_items=2, max_items=10)


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str = Field(min_length=1)
    criteria: None = None


Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


class DecisionRequest(BaseModel):
    model: str = "qwen3-14b-jev-style"
    state: Any
    questions: Dict[str, Question] = Field(min_items=1, max_items=128)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: Dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: Dict[str, float]
    confidence: float
    legend: List[str]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float
    probabilities: Dict[str, float]


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


class DecisionResponse(BaseModel):
    model: str
    answers: Dict[str, Answer]
    backend: Literal["qwen-logits"] = "qwen-logits"
