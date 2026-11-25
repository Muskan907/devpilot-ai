"""Pydantic schema for LLM-as-Judge verdicts.

Used for agent outputs that executable tests cannot score (synopsis Section 2.3):
plan quality, implementation quality, test adequacy and failure-diagnosis quality.
"""

from typing import Optional
from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    score: int = Field(..., ge=1, le=5, description="1 = very poor, 5 = excellent")
    justification: str = Field(..., description="One or two sentences citing concrete evidence")


class JudgeVerdict(BaseModel):
    plan_quality: CriterionScore = Field(..., description="Acceptance criteria are complete, testable and match the requirement")
    code_correctness: CriterionScore = Field(..., description="Implementation satisfies the requirement including edge cases")
    code_quality: CriterionScore = Field(..., description="Minimal, readable change consistent with the existing code")
    test_adequacy: CriterionScore = Field(..., description="Generated tests cover the acceptance criteria and edge cases")
    diagnosis_quality: Optional[CriterionScore] = Field(
        default=None, description="Only when a failure was diagnosed: is the root cause correct and actionable?"
    )
    overall: float = Field(default=0.0, ge=0.0, le=5.0)
    judge_model: str = ""

    def compute_overall(self) -> "JudgeVerdict":
        scores = [self.plan_quality.score, self.code_correctness.score, self.code_quality.score, self.test_adequacy.score]
        if self.diagnosis_quality:
            scores.append(self.diagnosis_quality.score)
        self.overall = round(sum(scores) / len(scores), 2)
        return self
