"""Critic-Reflection Loop Architecture Orchestrator.

Implements an iterative Generator-Critic multi-agent reflection loop:
1. Generator (Google ADK Coder) synthesizes candidate code changes.
2. Critic (Google ADK QA / Tester) evaluates the implementation by running pytest.
3. If defects or assertion errors occur, the Critic formulates a targeted critique.
4. Generator reflects on the critique, amends the implementation, and reapplies the patch.
5. Critic re-evaluates until all tests pass or max reflections are reached.
"""

import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

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


class CriticReflectionOrchestrator:
    """Iterative Generator-Critic reflection loop architecture."""

    def __init__(
        self,
        persona: Union[str, PersonaType, PersonaConfig] = PersonaType.BALANCED_ENGINEER,
        max_reflections: int = 2,
    ):
        self.pattern = AgenticPatternType.CRITIC_REFLECTION_LOOP
        self.persona_config = persona if isinstance(persona, PersonaConfig) else get_persona(persona)
        self.max_reflections = max_reflections
        self.generator = GoogleADKCoderAgent()
        self.critic = GoogleADKTestGeneratorAgent()

    def execute_requirement(
        self,
        raw_prompt: str,
        target_repo: Union[str, RepositoryConfig] = "sample_repo",
        auto_commit_if_approved: bool = False,
    ) -> PerformanceMetrics:
        """Executes the generator-critic reflection loop."""
        run_id = f"RUN-REFL-{uuid.uuid4().hex[:8]}"
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
            mcp_repo_client.create_branch(run_id=run_sandbox_id, task_id=repo_cfg.task_id or "TASK-REFL")

        # Baseline test run on untouched code (regression detection, same as C1/C2)
        baseline = mcp_test_client.run_tests(sandbox_path)
        if baseline.get("success") and baseline.get("report"):
            global_blackboard.update_metadata("baseline_test_report", baseline["report"])

        tool_calls = 2  # sandbox creation, baseline run
        reflection_cycles = 0

        # 3. Initial Generation by Generator
        gen_instruction = (
            f"Implement requirement: {raw_prompt}\n\n"
            f"[Engineering Persona Guidance: {self.persona_config.title}]\n"
            f"{self.persona_config.system_instruction_modifier}"
        )
        proposal = self.generator.implement_changes(
            run_id=run_sandbox_id,
            task_id="TSK-GEN-01",
            instruction=gen_instruction,
            context={"sandbox_path": sandbox_path},
        )
        tool_calls += 1  # write patch

        # 4. Critic generates targeted tests and runs the full suite
        report: TestResultReport = self.critic.generate_and_run_tests(
            run_id=run_sandbox_id,
            sandbox_path=sandbox_path,
            task_id="TSK-CRITIC-EVAL-0",
        )
        tool_calls += 2  # write tests, run pytest

        # 5. Reflection Loop: critique built from the real failure output, generator revises
        while not report.all_passed and reflection_cycles < self.max_reflections:
            reflection_cycles += 1
            failures = [
                f"{it.test_node_id}: {(it.failure_message or 'failed')[:300]}"
                for it in report.items
                if it.outcome in ("failed", "error")
            ]
            target_label = next(
                (p for p in global_blackboard.state.active_patches if "test" not in Path(p).name.lower()),
                "the target file",
            )
            critique = (
                f"CRITIQUE (Reflection Cycle {reflection_cycles}): "
                f"{report.failed + report.errors} failing test(s):\n" + "\n".join(failures[:10]) +
                f"\nReflect on these failures and correct the implementation in {target_label}."
            )

            proposal = self.generator.implement_changes(
                run_id=run_sandbox_id,
                task_id=f"TSK-REFL-GEN-{reflection_cycles}",
                instruction=f"{gen_instruction}\n\n[CRITIQUE FEEDBACK]:\n{critique}",
                context={"sandbox_path": sandbox_path, "critique": critique, "target_file": target_label},
            )
            tool_calls += 1

            report = self.critic.run_tests_only(
                run_id=run_sandbox_id,
                sandbox_path=sandbox_path,
                task_id=f"TSK-CRITIC-EVAL-{reflection_cycles}",
            )
            tool_calls += 1

        regressions = global_blackboard.state.metadata.get("regressions", [])

        # 6. Static AST scan over every file changed in this run
        patches_to_scan = [
            (path, patch.modified_content)
            for path, patch in global_blackboard.state.active_patches.items()
        ]
        audit = security_suite.run_full_audit(raw_prompt, patches_to_scan)

        # 7. HITL Gatekeeper
        hitl_gatekeeper.request_approval()
        if auto_commit_if_approved or settings.AUTO_APPROVE_IN_TESTS:
            hitl_gatekeeper.approve_and_commit(
                run_id=run_sandbox_id,
                approver="Automated Test / CI Suite",
                notes="Auto-approved in critic-reflection mode",
                target_repo=repo_cfg.repo_url or "sample_repo",
            )

        # 8. Metrics from the usage ledger (identical measurement method to C1/C2)
        total_time = round(time.time() - overall_start, 3)
        breakdowns = llm_breakdowns(settings.DEFAULT_MODEL)
        total_prompt_tokens = sum(b.prompt_tokens for b in breakdowns)
        total_comp_tokens = sum(b.completion_tokens for b in breakdowns)
        total_tokens = total_prompt_tokens + total_comp_tokens
        total_cost = round(sum(b.estimated_cost_usd for b in breakdowns), 8)
        pass_rate = (report.passed / report.total_tests) * 100.0 if report.total_tests > 0 else 0.0
        success = report.all_passed and audit.passed and not regressions

        return PerformanceMetrics(
            run_id=run_id,
            architecture_pattern=self.pattern.value,
            persona=self.persona_config.persona_type.value,
            benchmark_task=raw_prompt,
            total_execution_time_seconds=total_time,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_comp_tokens,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            test_pass_rate_percentage=round(pass_rate, 2),
            tests_passed=report.passed,
            tests_total=report.total_tests,
            failure_recovery_cycles=reflection_cycles,
            tool_call_count=tool_calls,
            a2a_message_count=0,  # Generator-Critic direct reflection loop (no A2A bus)
            agent_breakdowns=breakdowns,
            success=success,
            error_details=None if success else (
                f"Critic-reflection ended after {reflection_cycles} reflection(s): "
                f"tests_all_passed={report.all_passed}, regressions={len(regressions)}, audit_passed={audit.passed}"
            ),
            model_name=settings.DEFAULT_MODEL,
            is_measurement_estimated=usage_ledger.any_estimated(),
            quality={"coverage": mcp_test_client.run_coverage(sandbox_path)},
        )
