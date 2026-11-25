"""Python Client interface for MCP Test Runner Tool."""

from typing import Dict, Any
from mcp_servers.test_runner.server import check_python_syntax, run_pytest, measure_coverage
from core.schemas.tests import TestResultReport
from core.observability.tool_validation import (
    validated_tool, TestRunResult, SyntaxResult, CoverageResult,
)


class MCPTestRunnerClient:
    """Client wrapper for the MCP Test Execution Tool."""

    def __init__(self):
        self.name = "mcp-test-runner-client"

    @validated_tool("mcp_test_runner:check_python_syntax", SyntaxResult)
    def check_syntax(self, sandbox_path: str) -> Dict[str, Any]:
        return check_python_syntax(sandbox_path)

    @validated_tool("mcp_test_runner:run_pytest", TestRunResult)
    def run_tests(
        self,
        sandbox_path: str,
        test_rel_path: str = "tests",
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        result = run_pytest(
            sandbox_path=sandbox_path,
            test_rel_path=test_rel_path,
            timeout_seconds=timeout_seconds,
        )
        # Parse into typed TestResultReport if available
        if result.get("success") and result.get("report"):
            try:
                result["typed_report"] = TestResultReport.model_validate(result["report"])
            except Exception:
                result["typed_report"] = None
        return result

    @validated_tool("mcp_test_runner:measure_coverage", CoverageResult)
    def run_coverage(self, sandbox_path: str, source_rel_path: str = "src") -> Dict[str, Any]:
        """Line coverage of the source folder under the sandbox test suite (MCP tool)."""
        return measure_coverage(sandbox_path=sandbox_path, source_rel_path=source_rel_path)


# Global client instance
mcp_test_client = MCPTestRunnerClient()

