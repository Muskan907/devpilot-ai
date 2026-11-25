"""Tests for the evaluation extensions: P2P messaging (C3), coverage tool, retrieval metrics,
LLM-as-Judge parsing and per-model cost accounting. All offline (deterministic simulation)."""

import pytest

from config.settings import settings
from core.a2a import global_a2a_bus
from core.blackboard import global_blackboard
from core.observability import usage_ledger
from core.schemas.judge import JudgeVerdict
from evaluation.retrieval_metrics import retrieval_quality
from mcp_servers.repo_tool import mcp_repo_client
from mcp_servers.test_runner import mcp_test_client
from agents.google_adk.adk_runtime import parse_json_model

sim_only = pytest.mark.skipif(settings.EXECUTION_MODE == "live", reason="simulation-only test")


@sim_only
def test_p2p_runs_without_manager_and_uses_direct_messages():
    from orchestration.engine import AutonomousEngineeringOrchestrator
    from orchestration.patterns import AgenticPatternType

    m = AutonomousEngineeringOrchestrator(pattern=AgenticPatternType.HIERARCHICAL_P2P_A2A).execute_requirement(
        "Implement standard_deviation calculation in AdvancedCalculator"
    )
    assert m.architecture_pattern == "Hierarchical-P2P-A2A"
    assert m.a2a_message_count >= 3  # RAG->Planner, Planner->Coder, Coder->Tester
    senders = {e.sender_id for e in global_a2a_bus.get_history()}
    assert "P2P:Planner" in senders and "P2P:Coder" in senders
    assert global_blackboard.state.manager_decisions == []  # no Manager hub in P2P
    assert "coverage" in m.quality and m.quality["coverage"].get("success") is True


def test_coverage_tool_measures_sandbox_copy():
    sb = mcp_repo_client.create_sandbox("sample_repo")
    cov = mcp_test_client.run_coverage(sb["sandbox_path"])
    assert cov["success"] is True
    assert 50.0 <= cov["percent_covered"] <= 100.0


def test_retrieval_metrics_known_ranking():
    chunks = [
        {"file_path": "src/calculator.py", "symbol_name": "AdvancedCalculator.add"},
        {"file_path": "src/calculator.py", "symbol_name": "AdvancedCalculator.mean"},
        {"file_path": "src/__init__.py", "symbol_name": "module"},
    ]
    q = retrieval_quality(chunks, ["src/calculator.py"], ["mean", "AdvancedCalculator"])
    assert q["precision_at_k"] == round(1 / 3, 3)
    assert q["hit_at_1"] == 0.0
    assert q["mrr"] == 0.5
    assert q["recall"] == 0.5  # 'mean' found, bare class chunk not retrieved


def test_judge_verdict_parses_and_scores():
    raw = """Here is my evaluation:
```json
{"plan_quality": {"score": 4, "justification": "Criteria cover empty input."},
 "code_correctness": {"score": 5, "justification": "Handles even/odd lengths."},
 "code_quality": {"score": 4, "justification": "Minimal change."},
 "test_adequacy": {"score": 3, "justification": "No test for single element."},
 "diagnosis_quality": null}
```"""
    v = parse_json_model(raw, JudgeVerdict).compute_overall()
    assert v.overall == 4.0
    assert v.diagnosis_quality is None


def test_local_model_calls_cost_nothing():
    usage_ledger.reset()
    usage_ledger.record("GoogleADK:CoderAgent", "google_adk", prompt_tokens=1000, completion_tokens=1000,
                        model="ollama_chat/qwen2.5:1.5b")
    usage_ledger.record("CrewAI:PlannerAgent", "crewai", prompt_tokens=1000, completion_tokens=1000,
                        model="gemini-3.5-flash")
    summary = usage_ledger.per_agent_summary()
    assert summary["GoogleADK:CoderAgent"]["cost_usd"] == 0.0
    assert summary["CrewAI:PlannerAgent"]["cost_usd"] > 0.0


def test_tool_output_validation_flags_malformed_results():
    from core.observability.tool_validation import (
        tool_validation_stats, validate_tool_result, TestRunResult, CoverageResult,
    )
    tool_validation_stats.reset()
    assert validate_tool_result("run_pytest", {"success": True}, TestRunResult) is False  # missing report
    assert validate_tool_result("coverage", {"success": True, "percent_covered": 140.0}, CoverageResult) is False
    assert validate_tool_result("coverage", {"success": True, "percent_covered": 88.5}, CoverageResult) is True
    snap = tool_validation_stats.snapshot()
    assert snap["checked"] == 3 and len(snap["violations"]) == 2


def test_coder_prompt_styles_configure_agent():
    from agents.google_adk import GoogleADKCoderAgent
    react = GoogleADKCoderAgent(prompt_style="react")
    assert {t.__name__ for t in react.adk_agent.tools} == {"list_repo_files", "read_repo_file"}
    assert "ReAct loop" in react.adk_agent.instruction
    assert not GoogleADKCoderAgent(prompt_style="few_shot").adk_agent.tools
    with pytest.raises(ValueError):
        GoogleADKCoderAgent(prompt_style="chain_of_nonsense")
