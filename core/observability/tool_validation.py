"""Callback-based validation of MCP tool return values (requirement 8: "callbacks for data
validation, e.g. type checking from a tool's return").

Every MCP client method is wrapped by `validated_tool(name)`, which:
  1. fires on_tool_start,
  2. calls the tool,
  3. type-checks the return value against a Pydantic schema for that tool,
  4. fires on_tool_end, and on_guardrail_check("ToolOutputValidation", ...) with the verdict.
A malformed result is logged as a guardrail violation and counted; the raw result is still
returned so the caller's own error handling decides what to do (no silent coercion).
"""

import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from core.observability.callbacks import callback_manager
from core.schemas.tests import TestResultReport


class _ToolResult(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    success: bool
    error: Optional[str] = None


class SandboxResult(_ToolResult):
    run_id: Optional[str] = None
    sandbox_path: Optional[str] = None

    @model_validator(mode="after")
    def _paths_when_ok(self):
        if self.success and not (self.run_id and self.sandbox_path):
            raise ValueError("successful sandbox result must include run_id and sandbox_path")
        return self


class ReadFileResult(_ToolResult):
    content: Optional[str] = None

    @model_validator(mode="after")
    def _content_when_ok(self):
        if self.success and not isinstance(self.content, str):
            raise ValueError("successful read must return string content")
        return self


class ListFilesResult(_ToolResult):
    files: List[str] = Field(default_factory=list)


class PatchResult(_ToolResult):
    diff: Optional[str] = None


class TestRunResult(_ToolResult):
    __test__ = False
    report: Optional[TestResultReport] = None

    @model_validator(mode="after")
    def _report_when_ok(self):
        if self.success and self.report is None:
            raise ValueError("successful test run must include a structured report")
        return self


class SyntaxResult(_ToolResult):
    syntax_errors: Optional[list] = None


class CoverageResult(_ToolResult):
    percent_covered: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class _ValidationStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.checked = 0
        self.violations: List[Dict[str, str]] = []

    def record(self, tool: str, ok: bool, details: str) -> None:
        with self._lock:
            self.checked += 1
            if not ok:
                self.violations.append({"tool": tool, "details": details})

    def reset(self) -> None:
        with self._lock:
            self.checked = 0
            self.violations = []

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"checked": self.checked, "violations": list(self.violations)}


tool_validation_stats = _ValidationStats()


def validate_tool_result(tool_name: str, result: Any, schema: Type[BaseModel]) -> bool:
    """Type-checks one tool result and emits the guardrail callback. Returns True if valid."""
    if not isinstance(result, dict):
        details = f"{tool_name} returned {type(result).__name__}, expected dict"
        ok = False
    else:
        try:
            schema.model_validate(result)
            ok, details = True, f"{tool_name} output matches {schema.__name__}"
        except ValidationError as err:
            ok = False
            details = f"{tool_name} output failed {schema.__name__}: {err.errors()[0].get('msg', str(err))[:160]}"
    tool_validation_stats.record(tool_name, ok, details)
    callback_manager.on_guardrail_check("ToolOutputValidation", ok, details)
    return ok


def validated_tool(tool_name: str, schema: Type[BaseModel]) -> Callable:
    """Decorator for MCP client methods: start/end callbacks + return-type validation."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            callback_manager.on_tool_start(tool_name, {k: str(v)[:80] for k, v in kwargs.items()})
            result = fn(*args, **kwargs)
            validate_tool_result(tool_name, result, schema)
            summary = "success" if isinstance(result, dict) and result.get("success") else "failure"
            callback_manager.on_tool_end(tool_name, summary)
            return result

        return wrapper

    return decorator
