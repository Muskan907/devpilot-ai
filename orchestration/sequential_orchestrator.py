"""Sequential Baseline Architecture Orchestrator.

A monolithic single-agent baseline executing requirements sequentially in a single context:
- No reactive Shared Blackboard (zero blackboard state events).
- No A2A Protocol bus (zero inter-agent envelopes).
- No specialized role delegation (single-pass linear pipeline).
- No failure-driven replanning loop.
"""

import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Union

from config.settings import settings
from config.models_config import calculate_cost
from core.schemas import PerformanceMetrics, AgentTokenUsage, TestResultReport, RepositoryConfig
from core.blackboard import global_blackboard
from core.guardrails import security_suite
from core.hitl import hitl_gatekeeper
from mcp_servers.repo_tool import mcp_repo_client
from mcp_servers.test_runner import mcp_test_client
from agents.google_adk import GoogleADKCoderAgent, GoogleADKTestGeneratorAgent
from core.observability import usage_ledger
from core.a2a import global_a2a_bus
from evaluation.metrics_collector import llm_breakdowns
from orchestration.personas import PersonaType, get_persona, PersonaConfig
from orchestration.patterns import AgenticPatternType


class SequentialBaselineOrchestrator:
    """Monolithic single-agent sequential baseline architecture."""

    def __init__(self, persona: Union[str, PersonaType, PersonaConfig] = PersonaType.BALANCED_ENGINEER):
        self.pattern = AgenticPatternType.SEQUENTIAL_BASELINE
        self.persona_config = persona if isinstance(persona, PersonaConfig) else get_persona(persona)
        self.coder = GoogleADKCoderAgent()
        self.tester = GoogleADKTestGeneratorAgent()

    def execute_requirement(
        self,
        raw_prompt: str,
        target_repo: Union[str, RepositoryConfig] = "sample_repo",
        auto_commit_if_approved: bool = False,
    ) -> PerformanceMetrics:
        """Executes a single-pass linear pipeline without blackboard or A2A delegation."""
        run_id = f"RUN-SEQ-{uuid.uuid4().hex[:8]}"
        overall_start = time.time()
        global_blackboard.reset()
        usage_ledger.reset()
        global_a2a_bus.clear()

        # 1. Guardrail Check
        prompt_ok, prompt_violations = security_suite.prompt_guard.validate_prompt(raw_prompt)
        if not prompt_ok:
            return PerformanceMetrics(
                run_id=run_id,
                architecture_pattern=self.pattern.value,
                persona=self.persona_config.persona_type.value,
                benchmark_task=raw_prompt,
                total_execution_time_seconds=round(time.time() - overall_start, 3),
                success=False,
                error_details=f"Prompt rejected: {prompt_violations[0].details}",
            )

        # 2. Normalize repository configuration and create isolated sandbox
        if isinstance(target_repo, RepositoryConfig):
            repo_cfg = target_repo
        elif isinstance(target_repo, str):
            if target_repo.startswith("https://") or target_repo.startswith("http://"):
                repo_cfg = RepositoryConfig(repo_url=target_repo)
            else:
                repo_cfg = RepositoryConfig(repo_url=target_repo, base_branch="main")
        else:
            repo_cfg = RepositoryConfig()

        is_git_target = repo_cfg.is_github or (
            repo_cfg.repo_url
            and Path(repo_cfg.repo_url).exists()
            and (Path(repo_cfg.repo_url) / ".git").exists()
        )

        if is_git_target:
            sandbox_res = mcp_repo_client.clone_repository(
                repo_url=repo_cfg.repo_url,
                base_branch=repo_cfg.base_branch,
            )
        else:
            sandbox_res = mcp_repo_client.create_sandbox(repo_cfg.repo_url)

        if not sandbox_res.get("success"):
            return PerformanceMetrics(
                run_id=run_id,
                architecture_pattern=self.pattern.value,
                persona=self.persona_config.persona_type.value,
                benchmark_task=raw_prompt,
                total_execution_time_seconds=round(time.time() - overall_start, 3),
                success=False,
                error_details=f"Sandbox creation failed: {sandbox_res.get('error')}",
            )

        run_sandbox_id = sandbox_res["run_id"]
        sandbox_path = sandbox_res["sandbox_path"]
        sandbox_dir = Path(sandbox_path)

        if (sandbox_dir / ".git").exists():
            mcp_repo_client.create_branch(run_id=run_sandbox_id, task_id=repo_cfg.task_id or "TASK-SEQ")

        # Baseline test run on untouched code (regression detection, same as C1)
        baseline = mcp_test_client.run_tests(sandbox_path)
        if baseline.get("success") and baseline.get("report"):
            global_blackboard.update_metadata("baseline_test_report", baseline["report"])

        # 3. Monolithic code generation
        instruction = (
            f"{raw_prompt}\n\n"
            f"[Engineering Persona Guidance: {self.persona_config.title}]\n"
            f"{self.persona_config.system_instruction_modifier}"
        )

        proposal = self.coder.implement_changes(
            run_id=run_sandbox_id,
            task_id="TSK-SEQ-01",
            instruction=instruction,
            context={"sandbox_path": sandbox_path},
        )

        # 4-5. Same Testing agent as C1: targeted test generation + full-suite run,
        #      but a single pass (no Manager, no debugger, no re-planning loop).
        test_report = self.tester.generate_and_run_tests(
            run_id=run_sandbox_id,
            sandbox_path=sandbox_path,
            task_id="TSK-SEQ-02-TEST",
        )
        regressions = global_blackboard.state.metadata.get("regressions", [])

        # 6. Static AST scan
        patches_to_scan = []
        for patch in proposal.patches:
            patches_to_scan.append((patch.file_path, patch.modified_content))
        audit = security_suite.run_full_audit(raw_prompt, patches_to_scan)

        # 7. HITL Gatekeeper
        hitl_gatekeeper.request_approval()
        if auto_commit_if_approved or settings.AUTO_APPROVE_IN_TESTS:
            hitl_gatekeeper.approve_and_commit(
                run_id=run_sandbox_id,
                approver="Automated Test / CI Suite",
                notes="Auto-approved in sequential testing mode",
                target_repo=repo_cfg.repo_url or "sample_repo",
            )

        # 8. Metrics from the usage ledger (identical measurement method to C1/C3)
        total_time = round(time.time() - overall_start, 3)
        breakdowns = llm_breakdowns(settings.DEFAULT_MODEL)
        prompt_tokens = sum(b.prompt_tokens for b in breakdowns)
        comp_tokens = sum(b.completion_tokens for b in breakdowns)
        total_tokens = prompt_tokens + comp_tokens
        total_cost = round(sum(b.estimated_cost_usd for b in breakdowns), 8)
        success = test_report.all_passed and audit.passed and not regressions

        pass_rate = (test_report.passed / test_report.total_tests) * 100.0 if test_report.total_tests > 0 else 0.0

        metrics = PerformanceMetrics(
            run_id=run_id,
            architecture_pattern=self.pattern.value,
            persona=self.persona_config.persona_type.value,
            benchmark_task=raw_prompt,
            total_execution_time_seconds=total_time,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=comp_tokens,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            test_pass_rate_percentage=round(pass_rate, 2),
            tests_passed=test_report.passed,
            tests_total=test_report.total_tests,
            failure_recovery_cycles=0,  # Single-pass: no recovery loop
            tool_call_count=3,          # create_sandbox, apply_patch, run_tests
            a2a_message_count=0,        # No A2A communication
            agent_breakdowns=breakdowns,
            success=success,
            error_details=None if success else (
                f"Sequential single pass failed: tests_all_passed={test_report.all_passed}, "
                f"regressions={len(regressions)}, audit_passed={audit.passed}"
            ),
            model_name=settings.DEFAULT_MODEL,
            is_measurement_estimated=usage_ledger.any_estimated(),
            quality={"coverage": mcp_test_client.run_coverage(sandbox_path)},
        )

        return metrics

