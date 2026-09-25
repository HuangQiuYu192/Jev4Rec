"""Local, Qwen-backed typed decisions for Jev4Rec.

This package implements a Jev-compatible *interface*, not the hosted
Typesafe Jev model. Results produced by it must be reported as Qwen Jev-style.
"""

from .schemas import DecisionRequest, DecisionResponse

__all__ = ["DecisionRequest", "DecisionResponse"]
