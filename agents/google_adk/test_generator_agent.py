"""Google ADK Test Generator / Validation Agent.

Responsibilities (synopsis: Testing/Validation Agent):
  1. Generate targeted pytest tests for the change, from the acceptance criteria and the
     code the Coder actually wrote (real Gemini call through the ADK Runner in live mode).
  2. Run the targeted tests together with the repository's existing suite via the MCP
     Test Runner tool.
  3. Detect regressions: tests that passed on the untouched baseline but fail now.
"""

import ast
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from google.adk.agents import LlmAgent

from config.settings import settings
from core.schemas import (
    TestResultReport,
    TestResultItem,
    TaskStatus,
    CodeChangeProposal,
    FilePatch,
    ChangeType,
)
from core.a2a import A2AEnvelope, A2AMessageType, global_a2a_bus
from core.blackboard import global_blackboard
from core.observability import callback_manager
from mcp_servers.repo_tool import mcp_repo_client
from mcp_servers.test_runner import mcp_test_client
from agents.google_adk.adk_runtime import ADKInvoker, extract_code_block, record_simulated_call

GENERATED_TEST_FILE = "tests/test_devpilot_generated.py"

TEST_GEN_INSTRUCTION = (
    "You are an expert QA engineer. You write focused, deterministic pytest test files "
    "that verify a specific code change against its acceptance criteria. Use plain asserts "
    "and pytest.raises for error cases, cover normal cases plus boundary/edge cases, never "
    "use network, randomness or sleeps, and never modify production code. Import the code "
    "under test exactly the way the repository's existing tests do. "
    "Output ONLY one complete Python file inside a single ```python ... ``` block."
)


class GoogleADKTestGeneratorAgent:
    """Google ADK-powered Testing / Validation Agent."""

    def __init__(self, model_name: Optional[str] = None):
        self.agent_id = "GoogleADK:TestGeneratorAgent"
        self.framework = "google_adk"
        self.model_name = model_name or settings.DEFAULT_MODEL
        self.last_token_usage: Optional[Dict[str, Any]] = None

        def execute_pytest(sandbox_path: str) -> dict:
            """Executes sandboxed pytest run via MCP."""
            return mcp_test_client.run_tests(sandbox_path)

        self.adk_agent = LlmAgent(
            name="google_adk_test_generator",
            description="Autonomous Software Engineering Test Generation and Verification Specialist",
            instruction=TEST_GEN_INSTRUCTION,
            model=self.model_name,
        )
        self.invoker = ADKInvoker(self.adk_agent, self.agent_id)
        # Kept for tool-use demonstrations; generation itself does not need tools.
        self.execute_pytest_tool = execute_pytest

        global_a2a_bus.register_agent(self.agent_id, self.handle_a2a_message)

    # ------------------------------------------------------------------ A2A
    def handle_a2a_message(self, envelope: A2AEnvelope) -> Optional[A2AEnvelope]:
        """Handles incoming A2A delegation from CrewAI Manager."""
        if envelope.message_type == A2AMessageType.TASK_DELEGATION:
            task_id = envelope.payload.get("task_id", "TSK-03-TESTING")
            run_id = envelope.payload.get("run_id")
            sandbox_path = envelope.payload.get("context", {}).get("sandbox_path")

            report = self.generate_and_run_tests(run_id=run_id, sandbox_path=sandbox_path, task_id=task_id)
            regressions = global_blackboard.state.metadata.get("regressions", [])

            return A2AEnvelope(
                sender_id=self.agent_id,
                sender_framework=self.framework,
                recipient_id=envelope.sender_id,
                recipient_framework=envelope.sender_framework,
                correlation_id=envelope.message_id,
                message_type=A2AMessageType.TASK_RESULT,
                payload={
                    "task_id": task_id,
                    "status": "completed" if report.all_passed else "failed",
                    "all_passed": report.all_passed,
                    "passed": report.passed,
                    "failed": report.failed,
                    "total": report.total_tests,
                    "regressions": regressions,
                    "summary": (
                        f"Pytest completed: {report.passed}/{report.total_tests} passed, "
                        f"{len(regressions)} regression(s)"
                    ),
                    "token_usage": self.last_token_usage,
                },
            )
        return None

    # --------------------------------------------------------------- public
    def generate_and_run_tests(
        self,
        run_id: str,
        sandbox_path: str,
        task_id: str = "TSK-03-TESTING",
    ) -> TestResultReport:
        """Generates targeted tests for the change (if applicable), then runs the full suite."""
        sb_dir = Path(sandbox_path).resolve()
        test_candidates = (
            list(sb_dir.glob("tests/**/test_*.py"))
            + list(sb_dir.glob("test/**/test_*.py"))
            + list(sb_dir.glob("test_*.py"))
        )

        if test_candidates:
            if settings.EXECUTION_MODE == "live":
                self._generate_tests_live(run_id, sb_dir, test_candidates, task_id)
            else:
                self._generate_tests_simulated(run_id, sb_dir, test_candidates)

        return self.run_tests_only(run_id=run_id, sandbox_path=sandbox_path, task_id=task_id)

    def run_tests_only(
        self,
        run_id: str,
        sandbox_path: str,
        task_id: str = "TSK-03-TESTING",
    ) -> TestResultReport:
        """Runs syntax check + full test suite, computes regressions, posts the report."""
        sb_dir = Path(sandbox_path).resolve()
        report: Optional[TestResultReport] = None
        if list(sb_dir.rglob("*.py")):
            syntax_res = mcp_test_client.check_syntax(sandbox_path)
            if not syntax_res.get("success"):
                # Invalid generated code is an observable failure for the repair loop,
                # not a crash: report each syntax error as an errored check.
                errors = syntax_res.get("syntax_errors") or [{"file": "?", "msg": str(syntax_res.get("error"))}]
                report = TestResultReport(
                    total_tests=len(errors),
                    errors=len(errors),
                    all_passed=False,
                    items=[
                        TestResultItem(
                            test_node_id=f"syntax_check::{str(e.get('file', '?')).replace(chr(92), '/')}",
                            outcome="error",
                            failure_message=(
                                f"SyntaxError: {e.get('msg')} at line {e.get('line')}: {str(e.get('text', '')).strip()}"
                            ),
                        )
                        for e in errors
                    ],
                    raw_stdout=f"SyntaxError in sandbox: {errors}",
                )

        if report is None:
            test_run = mcp_test_client.run_tests(sandbox_path)
            if not test_run.get("success"):
                raise RuntimeError(f"Test execution failed: {test_run.get('error')}")
            report = TestResultReport.model_validate(test_run["report"])
        regressions = self.compute_regressions(report)
        global_blackboard.update_metadata("regressions", regressions)

        global_blackboard.post_test_report(report)
        global_blackboard.update_task_status(
            task_id=task_id,
            status=TaskStatus.COMPLETED if report.all_passed else TaskStatus.FAILED,
            output_summary=(
                f"Tests evaluated: {report.passed}/{report.total_tests} passed "
                f"(All Passed: {report.all_passed}); regressions: {len(regressions)}"
            ),
        )
        return report

    @staticmethod
    def compute_regressions(report: TestResultReport) -> List[str]:
        """Tests that passed on the baseline (pre-change) run but do not pass now."""
        baseline = global_blackboard.state.metadata.get("baseline_test_report")
        if not baseline:
            return []
        baseline_passed = {
            item.get("test_node_id")
            for item in baseline.get("items", [])
            if item.get("outcome") == "passed"
        }
        return sorted(
            item.test_node_id
            for item in report.items
            if item.test_node_id in baseline_passed and item.outcome in ("failed", "error")
        )

    # --------------------------------------------------------------- live
    def _generate_tests_live(
        self,
        run_id: str,
        sb_dir: Path,
        test_candidates: List[Path],
        task_id: str,
    ) -> None:
        changed_sources = {
            path: patch.modified_content
            for path, patch in global_blackboard.state.active_patches.items()
            if path.endswith(".py") and "test" not in Path(path).name.lower()
        }
        if not changed_sources:
            return  # Nothing new to test; existing suite still runs.

        example_test_file = str(test_candidates[0].relative_to(sb_dir)).replace("\\", "/")
        example_tests = mcp_repo_client.read_file(run_id, example_test_file).get("content", "")
        prompt = self._build_prompt(changed_sources, example_test_file, example_tests)

        callback_manager.on_agent_start(self.agent_id, self.framework, f"Generating targeted tests with {self.model_name}")
        response = self.invoker.invoke(prompt)
        self.last_token_usage = self.invoker.last_usage

        test_code = extract_code_block(response) or response
        try:
            ast.parse(test_code)
        except SyntaxError as err:
            callback_manager.on_agent_error(
                self.agent_id, self.framework, f"Generated tests are not valid Python ({err}); skipped writing them"
            )
            return

        self._write_generated_tests(run_id, test_code, task_id)

    def _build_prompt(self, changed_sources: Dict[str, str], example_file: str, example_tests: str) -> str:
        state = global_blackboard.state
        requirement = state.requirement.raw_prompt if state.requirement else ""
        criteria = ""
        if state.analysis:
            criteria = "\n".join(f"- {ac.id}: {ac.description}" for ac in state.analysis.acceptance_criteria)
        sources = "\n\n".join(
            f"File: {path}\n```python\n{code}\n```" for path, code in changed_sources.items()
        )
        return (
            f"Requirement: {requirement}\n\n"
            f"Acceptance criteria:\n{criteria or '- Implement the requirement correctly.'}\n\n"
            f"Code after the change:\n{sources}\n\n"
            f"Existing test file ({example_file}) - follow its import style and fixtures:\n"
            f"```python\n{example_tests}\n```\n\n"
            f"Write a NEW test file saved as {GENERATED_TEST_FILE} that verifies the requirement "
            f"and each acceptance criterion. Do not repeat the existing tests. If you use fixtures "
            f"from the existing file, redefine them in the new file (it is a separate module)."
        )

    def _write_generated_tests(self, run_id: str, test_code: str, task_id: str) -> None:
        existing = mcp_repo_client.read_file(run_id, GENERATED_TEST_FILE)
        original = existing.get("content", "") if existing.get("success") else ""
        patch_res = mcp_repo_client.apply_patch(
            run_id=run_id,
            file_path=GENERATED_TEST_FILE,
            content=test_code,
            justification="Targeted tests generated from acceptance criteria by the Test Generator agent.",
        )
        proposal = CodeChangeProposal(
            task_id=task_id,
            author_agent=self.agent_id,
            framework=self.framework,
            patches=[
                FilePatch(
                    file_path=GENERATED_TEST_FILE,
                    change_type=ChangeType.MODIFY if original else ChangeType.CREATE,
                    original_content=original,
                    modified_content=test_code,
                    unified_diff=patch_res.get("diff", ""),
                    affected_symbols=["generated_tests"],
                    justification="Generated targeted tests",
                )
            ],
        )
        global_blackboard.post_code_proposal(proposal)

    # ---------------------------------------------------------- simulation
    def _generate_tests_simulated(self, run_id: str, sb_dir: Path, test_candidates: List[Path]) -> None:
        """Deterministic offline path used by the automated test suite (no LLM)."""
        start = time.perf_counter()
        test_file = str(test_candidates[0].relative_to(sb_dir)).replace("\\", "/")
        original_tests = mcp_repo_client.read_file(run_id, test_file).get("content", "")
        new_tests = ""

        req_prompt = getattr(global_blackboard.state.analysis, "problem_statement", "") if global_blackboard.state.analysis else ""
        has_std_patch = any(
            "standard_deviation" in p.modified_content for p in global_blackboard.state.active_patches.values()
        )
        if (
            "calc" in original_tests
            and (has_std_patch or "standard_deviation" in req_prompt or "AdvancedCalculator" in req_prompt)
            and "def test_standard_deviation" not in original_tests
        ):
            new_tests = (
                "\n"
                "def test_standard_deviation_population(calc):\n"
                "    data = [2, 4, 4, 4, 5, 5, 7, 9]\n"
                "    assert calc.standard_deviation(data, sample=False) == 2.0\n"
                "\n"
                "def test_standard_deviation_sample(calc):\n"
                "    data = [10, 12, 23, 23, 16, 23, 21, 16]\n"
                "    std = calc.standard_deviation(data, sample=True)\n"
                "    assert round(std, 2) == 5.24\n"
                "\n"
                "def test_standard_deviation_empty_raises(calc):\n"
                "    with pytest.raises(ValueError):\n"
                "        calc.standard_deviation([])\n"
                "\n"
                "def test_standard_deviation_single_element_sample_raises(calc):\n"
                "    with pytest.raises(ValueError):\n"
                "        calc.standard_deviation([42], sample=True)\n"
            )
            mcp_repo_client.apply_patch(
                run_id=run_id,
                file_path=test_file,
                content=original_tests.rstrip() + new_tests + "\n",
                justification="[simulation] Added unit tests covering standard deviation and edge cases.",
            )
        self.last_token_usage = record_simulated_call(
            self.agent_id, self.framework, original_tests, new_tests, time.perf_counter() - start
        )
