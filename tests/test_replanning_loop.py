"""Tests for the failure-driven re-planning loop, regression detection and honest metrics.

Runs in deterministic simulation mode (no LLM calls): the Debugger uses its rule-based
classifier and the Manager follows the Debugger's recommendation, with hard policy rules.
"""

import pytest

from config.settings import settings
from core.blackboard import global_blackboard
from core.observability import usage_ledger, LLMQuotaExhaustedError
from core.schemas import CodeChangeProposal, FilePatch, ChangeType, RecoveryAction, FailureCategory
from mcp_servers.repo_tool import mcp_repo_client
from mcp_servers.test_runner import mcp_test_client
from agents.google_adk import GoogleADKTestGeneratorAgent
from orchestration.failure_recovery import FailureRecoveryController

pytestmark = pytest.mark.skipif(
    settings.EXECUTION_MODE == "live", reason="Loop tests are deterministic-simulation only"
)


@pytest.fixture
def sandbox():
    global_blackboard.reset()
    usage_ledger.reset()
    res = mcp_repo_client.create_sandbox("sample_repo")
    assert res["success"], res
    run_id, path = res["run_id"], res["sandbox_path"]
    baseline = mcp_test_client.run_tests(path)
    assert baseline["success"] and baseline["report"]["all_passed"]
    global_blackboard.update_metadata("baseline_test_report", baseline["report"])
    return run_id, path


def _write_source(run_id: str, new_code: str) -> None:
    original = mcp_repo_client.read_file(run_id, "src/calculator.py")["content"]
    res = mcp_repo_client.apply_patch(run_id=run_id, file_path="src/calculator.py", content=new_code,
                                      justification="test setup")
    global_blackboard.post_code_proposal(CodeChangeProposal(
        task_id="TSK-02-CODING", author_agent="test", framework="google_adk",
        patches=[FilePatch(file_path="src/calculator.py", change_type=ChangeType.MODIFY,
                           original_content=original, modified_content=new_code,
                           unified_diff=res.get("diff", ""), affected_symbols=["calculator"],
                           justification="test setup")],
    ))


def test_missing_import_is_diagnosed_and_repaired(sandbox):
    run_id, path = sandbox
    code = mcp_repo_client.read_file(run_id, "src/calculator.py")["content"]
    broken = code.replace("import math\n", "", 1).replace(
        "return sum(numbers) / len(numbers)", "return math.fsum(numbers) / len(numbers)"
    )
    _write_source(run_id, broken)  # mean() now raises NameError: name 'math' is not defined

    tester = GoogleADKTestGeneratorAgent()
    report = tester.run_tests_only(run_id=run_id, sandbox_path=path)
    assert not report.all_passed
    assert any("test_mean" in r for r in global_blackboard.state.metadata["regressions"])

    controller = FailureRecoveryController(max_recovery_cycles=3, tester=tester)
    summary = controller.run_recovery_loop(run_id=run_id, sandbox_path=path, raw_prompt="Keep power() working")

    assert summary["final_all_passed"] is True
    assert summary["diagnoses"][0] == FailureCategory.MISSING_IMPORT.value
    assert summary["decisions"][0] == RecoveryAction.REQUEST_CODE_FIX.value
    assert summary["cycles"] == 1
    assert global_blackboard.state.metadata["regressions"] == []
    assert global_blackboard.state.plan is None or global_blackboard.state.plan.replan_count >= 0


def test_regression_detection_lists_previously_passing_tests(sandbox):
    run_id, path = sandbox
    code = mcp_repo_client.read_file(run_id, "src/calculator.py")["content"]
    _write_source(run_id, code.replace("return a + b", "return a - b"))

    report = GoogleADKTestGeneratorAgent().run_tests_only(run_id=run_id, sandbox_path=path)
    regressions = global_blackboard.state.metadata["regressions"]
    assert not report.all_passed
    assert regressions == ["tests/test_calculator.py::test_add"] or any("test_add" in r for r in regressions)


def test_unfixable_failure_stops_within_retry_limit(sandbox):
    run_id, path = sandbox
    code = mcp_repo_client.read_file(run_id, "src/calculator.py")["content"]
    _write_source(run_id, code.replace("return a + b", "return a - b"))  # logic bug the sim coder cannot fix

    tester = GoogleADKTestGeneratorAgent()
    tester.run_tests_only(run_id=run_id, sandbox_path=path)
    controller = FailureRecoveryController(max_recovery_cycles=3, tester=tester)
    summary = controller.run_recovery_loop(run_id=run_id, sandbox_path=path, raw_prompt="Fix add")

    assert summary["final_all_passed"] is False
    assert summary["cycles"] <= 3
    assert summary["stop_reason"] != "tests_passed"
    decisions = global_blackboard.state.manager_decisions
    assert decisions[-1].action == RecoveryAction.STOP
    assert decisions[-1].overridden_by_policy is True


def test_quota_exhaustion_records_failed_estimated_run(monkeypatch):
    from orchestration.engine import AutonomousEngineeringOrchestrator

    orch = AutonomousEngineeringOrchestrator()

    def _boom(*args, **kwargs):
        raise LLMQuotaExhaustedError("CrewAI:PlannerAgent: LLM quota exhausted: 429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(orch.planner, "plan_requirement", _boom)
    metrics = orch.execute_requirement("Add a median method to AdvancedCalculator")

    assert metrics.success is False
    assert metrics.error_details.startswith("LLM_QUOTA_EXHAUSTED")
    assert metrics.is_measurement_estimated is True


def test_metrics_come_from_ledger_not_constants():
    from orchestration.engine import AutonomousEngineeringOrchestrator

    metrics = AutonomousEngineeringOrchestrator().execute_requirement(
        "Implement standard_deviation calculation in AdvancedCalculator"
    )
    by_agent = {b.agent_id: b for b in metrics.agent_breakdowns}

    # RAG / Auditor do no LLM work: real latency, zero tokens (no fabricated numbers).
    assert by_agent["GoogleADK:RepositoryRAGAgent"].total_tokens == 0
    assert by_agent["GoogleADK:RepositoryRAGAgent"].latency_seconds > 0
    assert by_agent["CrewAI:AuditorAgent"].total_tokens == 0
    # Old hardcoded constants must be gone.
    assert (by_agent["CrewAI:PlannerAgent"].prompt_tokens, by_agent["CrewAI:PlannerAgent"].completion_tokens) != (850, 320)
    # Simulation runs are always flagged as estimated.
    assert metrics.is_measurement_estimated is True
    assert metrics.total_tokens == sum(b.total_tokens for b in metrics.agent_breakdowns)


def test_syntax_error_becomes_failing_report_not_crash(sandbox):
    """Invalid generated code must be observable by the repair loop, not abort the run."""
    run_id, path = sandbox
    code = mcp_repo_client.read_file(run_id, "src/calculator.py")["content"]
    _write_source(run_id, code + "\n    def broken(( -> None:\n        pass\n")

    report = GoogleADKTestGeneratorAgent().run_tests_only(run_id=run_id, sandbox_path=path)
    assert report.all_passed is False
    assert report.errors >= 1
    assert any(i.test_node_id.startswith("syntax_check::") for i in report.items)
    assert any("SyntaxError" in (i.failure_message or "") for i in report.items)
