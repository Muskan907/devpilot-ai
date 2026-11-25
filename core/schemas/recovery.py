"""Pydantic schemas for failure diagnosis and Manager re-planning decisions.

These implement the synopsis loop: Testing reports a failure -> the Debugging/Review
agent classifies it (FailureDiagnosis) -> the Manager decides what to do next
(ManagerDecision): fetch more repository context, request a code/test fix, or stop.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """Likely cause of a failing test run (synopsis Section 1.2)."""
    MISSING_IMPORT = "missing_import"
    WRONG_CONTEXT = "wrong_context"
    INCORRECT_LOGIC = "incorrect_logic"
    BROKEN_TEST = "broken_test"
    SYNTAX_ERROR = "syntax_error"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    """What the Manager can decide after a failure."""
    RETRIEVE_MORE_CONTEXT = "retrieve_more_context"
    REQUEST_CODE_FIX = "request_code_fix"
    REQUEST_TEST_FIX = "request_test_fix"
    STOP = "stop"


class FailureDiagnosis(BaseModel):
    """Structured output of the Debugging/Review agent."""
    category: FailureCategory = Field(..., description="Classified root-cause category")
    root_cause: str = Field(..., description="One or two sentences explaining why the tests fail")
    failing_tests: List[str] = Field(default_factory=list, description="Node ids of failing tests")
    target_file: Optional[str] = Field(default=None, description="File that should be changed to fix the failure")
    recommended_action: RecoveryAction = Field(..., description="Debugger's recommendation to the Manager")
    fixed_content: Optional[str] = Field(
        default=None,
        description="Complete corrected content of target_file, when a direct fix is proposed",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    cycle: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ManagerDecision(BaseModel):
    """Structured output of the Manager's re-planning step."""
    action: RecoveryAction
    rationale: str
    cycle: int = 0
    decided_by: str = Field(default="CrewAI:ManagerAgent")
    overridden_by_policy: bool = Field(
        default=False,
        description="True when a hard rule (retry limit, missing fix) replaced the LLM's choice",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
