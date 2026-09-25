"""Qwen causal-LM logits readout for bounded typed decisions.

The implementation deliberately does not call ``generate``. Each question is
rendered as a compact, fixed-label prompt, then probability mass is restricted
to the allowed one-token answer labels. This is a Jev-style baseline, not a
reproduction of the proprietary Jev model or its calibration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from .schemas import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    temperature: float = 1.0
    max_state_chars: int = 12_000


class QwenLogitDecisionEngine:
    """Lazy Qwen loader and constrained-token probability reader."""

    def __init__(self, config: EngineConfig):
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.config.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.config.device)
        adapter_path = os.environ.get("JEV_LORA_ADAPTER")
        if adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, adapter_path).to(self.config.device)
        self._model.eval()

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    def _token_ids(self, labels: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for label in labels:
            encoded = self.tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    f"label {label!r} is not a single tokenizer token; "
                    "use a different label scheme before running an experiment"
                )
            ids.append(encoded[0])
        if len(ids) != len(set(ids)):
            raise ValueError("decision labels collide in the tokenizer vocabulary")
        return ids

    def _state_text(self, state: object) -> str:
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
        if len(text) > self.config.max_state_chars:
            raise ValueError(
                f"state has {len(text)} characters, exceeding max_state_chars="
                f"{self.config.max_state_chars}; trim it explicitly rather than silently truncating"
            )
        return text

    @staticmethod
    def _prompt(state: str, question: Question, labels: Sequence[str]) -> str:
        criteria: object
        if isinstance(question, ChoiceQuestion):
            criteria = {label: description for label, description in zip(labels, question.criteria.values())}
        elif isinstance(question, ScoreQuestion):
            criteria = {label: description for label, description in zip(labels, question.criteria)}
        else:
            criteria = {labels[0]: "No", labels[1]: "Yes"}
        return (
            "You are a bounded decision function. Do not explain.\n"
            f"STATE:\n{state}\n\n"
            f"QUESTION:\n{question.instructions}\n\n"
            f"ALLOWED ANSWERS:\n{json.dumps(criteria, ensure_ascii=False)}\n\n"
            "Return exactly one allowed label, without spaces or punctuation.\n"
            "ANSWER:"
        )

    @staticmethod
    def _labels(question: Question) -> Tuple[List[str], List[str]]:
        if isinstance(question, ChoiceQuestion):
            keys = list(question.criteria)
            return list(LABELS[: len(keys)]), keys
        if isinstance(question, ScoreQuestion):
            values = [str(index) for index in range(len(question.criteria))]
            return list(LABELS[: len(values)]), values
        return ["N", "Y"], ["false", "true"]

    @torch.inference_mode()
    def decide(self, request: DecisionRequest) -> DecisionResponse:
        state = self._state_text(request.state)
        prepared = []
        for name, question in request.questions.items():
            labels, output_keys = self._labels(question)
            prepared.append((name, question, labels, output_keys, self._prompt(state, question, labels)))

        encoded = self.tokenizer(
            [entry[-1] for entry in prepared], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.config.device)
        logits = self.model(**encoded).logits[:, -1, :]
        answers: Dict[str, Answer] = {}
        for row, (name, question, labels, output_keys, _) in enumerate(prepared):
            label_ids = torch.tensor(self._token_ids(labels), device=logits.device)
            # Model weights/logits may be bfloat16. Compute the tiny bounded
            # softmax in float32: bfloat16 rounds high-confidence values to
            # exactly 1.0, destroying the ordering needed for reranking.
            distribution = torch.softmax(logits[row, label_ids].float() / self.config.temperature, dim=0)
            probabilities = {key: float(prob) for key, prob in zip(output_keys, distribution.tolist())}
            confidence = max(probabilities.values())
            if isinstance(question, ChoiceQuestion):
                choice = max(probabilities, key=probabilities.get)
                answers[name] = ChoiceAnswer(choice=choice, probabilities=probabilities, confidence=confidence)
            elif isinstance(question, ScoreQuestion):
                score = sum(index * probabilities[str(index)] for index in range(len(question.criteria)))
                answers[name] = ScoreAnswer(
                    score=score,
                    probabilities=probabilities,
                    confidence=confidence,
                    legend=question.criteria,
                )
            else:
                answers[name] = NoulAnswer(noul=probabilities["true"], probabilities=probabilities)
        return DecisionResponse(model=request.model, answers=answers)


def engine_from_environment() -> QwenLogitDecisionEngine:
    return QwenLogitDecisionEngine(
        EngineConfig(
            model_path=os.environ.get("JEV_QWEN_MODEL", "Qwen/Qwen3-14B"),
            device=os.environ.get("JEV_DEVICE", "cuda:0"),
            dtype=os.environ.get("JEV_DTYPE", "bfloat16"),
            temperature=float(os.environ.get("JEV_TEMPERATURE", "1.0")),
        )
    )
