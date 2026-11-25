"""Autonomous Software Engineering Orchestration Engine (C1: Hierarchical + Blackboard + A2A).

Implements the synopsis flow end to end:
  requirement -> guardrails -> sandbox/branch -> baseline tests -> RAG context ->
  Planner (CrewAI) -> Coder (ADK, via A2A) -> Tester (ADK, via A2A) ->
  failure-driven re-planning loop (Debugger ADK + Manager CrewAI + Planner) ->
  security audit -> Auditor (CrewAI) -> human approval gate -> metrics.

All per-agent tokens / latency come from the usage ledger (real provider data in live
mode, explicitly flagged estimates in deterministic simulation). A run that cannot reach
the LLM (quota exhausted) is recorded as failed, never as a successful measurement.
"""

import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

from config.settings import settings
from config.models_config import calculate_cost
from core.schemas import (
    DeveloperRequirement,
    RepositoryConfig,
    ExecutionPlan,
    CodeChangeProposal,
    TestResultReport,
    PerformanceMetrics,
    AgentTokenUsage,
    TaskStatus,
)
from core.blackboard import global_blackboard
from core.a2a import global_a2a_bus
from core.guardrails import security_suite
from core.observability import (
    callback_manager,
    provenance_tracker,
    system_tracer,
    usage_ledger,
    LLMQuotaExhaustedError,
)
from core.hitl import hitl_gatekeeper
from mcp_servers.repo_tool import mcp_repo_client
from mcp_servers.test_runner import mcp_test_client
from agents.crewai import (
    CrewAIPlannerAgent,
    CrewAIManagerAgent,
    CrewAIAuditorAgent,
)
from agents.google_adk import (
    GoogleADKCoderAgent,
    GoogleADKTestGeneratorAgent,
    GoogleADKDebuggerAgent,
    RepositoryRAGAgent,
)
from orchestration.failure_recovery import FailureRecoveryController
from orchestration.patterns import AgenticPatternType
from orchestration.personas import PersonaType, get_persona, PersonaConfig
from orchestration.sequential_orchestrator import SequentialBaselineOrchestrator
from orchestration.critic_reflection_orchestrator import CriticReflectionOrchestrator
from orchestration.p2p_orchestrator import PeerToPeerOrchestrator
from evaluation.metrics_collector import metrics_collector, build_agent_breakdowns


def run_metadata(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Structured per-run record persisted alongside the metrics row (metadata_json)."""
    state = global_blackboard.state
    meta: Dict[str, Any] = {
        "agent_breakdowns": [
            b.model_dump(mode="json") for b in build_agent_breakdowns(settings.DEFAULT_MODEL)
        ],
        "llm_calls": usage_ledger.to_json(),
        "execution_mode": settings.EXECUTION_MODE,
        "regressions": state.metadata.get("regressions", []),
        "replan_count": state.plan.replan_count if state.plan else 0,
        "failure_diagnoses": [
            d.model_dump(mode="json", exclude={"fixed_content"}) for d in state.failure_diagnoses
        ],
        "manager_decisions": [d.model_dump(mode="json") for d in state.manager_decisions],
        "human_approval_status": state.human_approval_status,
    }
    if extra:
        meta.update(extra)
    return meta


class AutonomousEngineeringOrchestrator:
    """The primary engine executing autonomous software engineering workflows."""

    def __init__(
        self,
        pattern: AgenticPatternType = AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A,
        persona: Union[str, PersonaType, PersonaConfig] = PersonaType.BALANCED_ENGINEER,
        max_recovery_cycles: int = 3,
        coder_prompt_style: Optional[str] = None,
    ):
        self.pattern = pattern
        self.coder_prompt_style = coder_prompt_style
        self.persona_config = persona if isinstance(persona, PersonaConfig) else get_persona(persona)
        if self.pattern == AgenticPatternType.SEQUENTIAL_BASELINE:
            self._delegate = SequentialBaselineOrchestrator(persona=self.persona_config)
        elif self.pattern == AgenticPatternType.CRITIC_REFLECTION_LOOP:
            self._delegate = CriticReflectionOrchestrator(persona=self.persona_config)
        elif self.pattern == AgenticPatternType.HIERARCHICAL_P2P_A2A:
            self._delegate = PeerToPeerOrchestrator(persona=self.persona_config, max_recovery_cycles=max_recovery_cycles)
        else:
            # C1, or C4 (same hierarchy; different Coder LLM and RAG embedding model)
            cross_model = self.pattern == AgenticPatternType.HIERARCHICAL_CROSS_MODEL
            self._delegate = None
            self.planner = CrewAIPlannerAgent()
            self.manager = CrewAIManagerAgent()
            self.auditor = CrewAIAuditorAgent()
            self.coder = GoogleADKCoderAgent(
                model_name=settings.C4_CODER_MODEL if cross_model else None,
                prompt_style=coder_prompt_style,
            )
            self.tester = GoogleADKTestGeneratorAgent()
            self.debugger = GoogleADKDebuggerAgent()
            self.rag_agent = RepositoryRAGAgent(embedding_backend="ollama" if cross_model else "hash")
            self.recovery_controller = FailureRecoveryController(
                max_recovery_cycles=max_recovery_cycles,
                debugger=self.debugger,
                planner=self.planner,
                manager=self.manager,
                coder=self.coder,
                tester=self.tester,
                rag_agent=self.rag_agent,
            )

    # ------------------------------------------------------------------
    def execute_requirement(
        self,
        raw_prompt: str,
        target_repo: Union[str, RepositoryConfig] = "sample_repo",
        auto_commit_if_approved: bool = False,
    ) -> PerformanceMetrics:
        """Runs the complete multi-agent engineering lifecycle."""
        if self._delegate is not None:
            usage_ledger.reset()
            metrics = self._delegate.execute_requirement(
                raw_prompt=raw_prompt,
                target_repo=target_repo,
                auto_commit_if_approved=auto_commit_if_approved,
            )
            metrics_collector.record_run(metrics, benchmark_task=raw_prompt, persona=self.persona_config.persona_type.value)
            return metrics

        run_id = f"RUN-{uuid.uuid4().hex[:8]}"
        overall_start = time.time()
        callback_manager.on_pipeline_start(run_id, raw_prompt)

        # Reset session state
        global_blackboard.reset()
        provenance_tracker.clear()
        system_tracer.clear()
        usage_ledger.reset()
        global_a2a_bus.clear()

        try:
            return self._run_hierarchical(run_id, overall_start, raw_prompt, target_repo, auto_commit_if_approved)
        except LLMQuotaExhaustedError as exc:
            callback_manager.on_agent_error("Orchestrator", "orchestration", f"Run aborted: {exc}")
            metrics = self._finalize_metrics(
                run_id=run_id,
                overall_start=overall_start,
                raw_prompt=raw_prompt,
                success=False,
                error=f"LLM_QUOTA_EXHAUSTED: {str(exc)[:300]}",
                force_estimated=True,
            )
            return metrics

    # ------------------------------------------------------------------
    def _run_hierarchical(
        self,
        run_id: str,
        overall_start: float,
        raw_prompt: str,
        target_repo: Union[str, RepositoryConfig],
        auto_commit_if_approved: bool,
    ) -> PerformanceMetrics:
        # -------------------------------------------------------------
        # STEP 1: Guardrail Check (Prompt Injection & Input Validation)
        # -------------------------------------------------------------
        span_guard = system_tracer.start_span("guardrails_prompt_validation")
        prompt_ok, prompt_violations = security_suite.prompt_guard.validate_prompt(raw_prompt)
        span_guard.finish(passed=str(prompt_ok))

        if not prompt_ok:
            error_msg = f"Security Violation: Prompt Injection / Malicious Input detected: {prompt_violations[0].details}"
            callback_manager.on_guardrail_check("PromptGuardrail", False, error_msg)
            return PerformanceMetrics(
                run_id=run_id,
                architecture_pattern=self.pattern.value,
                total_execution_time_seconds=round(time.time() - overall_start, 3),
                success=False,
                error_details=error_msg,
            )
        callback_manager.on_guardrail_check("PromptGuardrail", True, "Prompt sanitized and approved")

        # -------------------------------------------------------------
        # STEP 2: Ephemeral Sandbox Creation / Git Clone via MCP Repo Tool
        # -------------------------------------------------------------
        span_mcp = system_tracer.start_span("mcp_create_isolated_sandbox")

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
            callback_manager.on_tool_start(
                "mcp_repo_tool:clone_github_repository",
                {"repo_url": repo_cfg.repo_url, "base_branch": repo_cfg.base_branch},
            )
            sandbox_res = mcp_repo_client.clone_repository(
                repo_url=repo_cfg.repo_url,
                base_branch=repo_cfg.base_branch,
            )
            span_mcp.finish(success=str(sandbox_res.get("success")))
            callback_manager.on_tool_end(
                "mcp_repo_tool:clone_github_repository",
                f"Cloned repository to {sandbox_res.get('sandbox_path')}",
            )
        else:
            callback_manager.on_tool_start(
                "mcp_repo_tool:create_isolated_sandbox",
                {"source_repo": repo_cfg.repo_url},
            )
            sandbox_res = mcp_repo_client.create_sandbox(repo_cfg.repo_url)
            span_mcp.finish(success=str(sandbox_res.get("success")))
            callback_manager.on_tool_end(
                "mcp_repo_tool:create_isolated_sandbox",
                f"Created sandbox at {sandbox_res.get('sandbox_path')}",
            )

        if not sandbox_res.get("success"):
            raise RuntimeError(f"Sandbox isolation failed: {sandbox_res.get('error')}")

        run_sandbox_id = sandbox_res["run_id"]
        sandbox_path = sandbox_res["sandbox_path"]
        sandbox_dir = Path(sandbox_path)

        # Create isolated agentic branch if it is a Git repository
        agentic_branch = None
        if (sandbox_dir / ".git").exists():
            branch_res = mcp_repo_client.create_branch(run_id=run_sandbox_id, task_id=repo_cfg.task_id or "TASK-01")
            if branch_res.get("success"):
                agentic_branch = branch_res.get("branch_name")

        global_blackboard.update_metadata("repository", {
            "repo_url": repo_cfg.repo_url,
            "is_github": repo_cfg.is_github,
            "base_branch": repo_cfg.base_branch,
            "agentic_branch": agentic_branch,
            "sandbox_path": sandbox_path,
            "run_id": run_sandbox_id,
        })

        # -------------------------------------------------------------
        # STEP 2.2: Baseline test run on the untouched code (for regression detection)
        # -------------------------------------------------------------
        span_base = system_tracer.start_span("baseline_test_run")
        with usage_ledger.timed_step("MCP:TestRunner", "mcp", note="baseline test run"):
            baseline = mcp_test_client.run_tests(sandbox_path)
        if baseline.get("success") and baseline.get("report"):
            global_blackboard.update_metadata("baseline_test_report", baseline["report"])
            span_base.finish(passed=str(baseline["report"].get("passed")), total=str(baseline["report"].get("total_tests")))
        else:
            span_base.finish(error=str(baseline.get("error")))

        # -------------------------------------------------------------
        # STEP 2.5: Repository RAG Intelligence (AST Chunker & Retrieval)
        # -------------------------------------------------------------
        span_rag = system_tracer.start_span("repository_rag_indexing_and_retrieval")
        rag_context = ""
        retrieved_chunks = []
        try:
            callback_manager.on_agent_start(
                self.rag_agent.agent_id,
                self.rag_agent.framework,
                "Repository AST Indexing and Context Retrieval",
            )
            with usage_ledger.timed_step(self.rag_agent.agent_id, self.rag_agent.framework, note="index + retrieve"):
                rag_res = self.rag_agent.retrieve_context(
                    query=raw_prompt,
                    repo_path=sandbox_dir,
                    repo_id=run_sandbox_id,
                    top_k=5,
                )
            rag_context = rag_res.get("context", "")
            retrieved_chunks = rag_res.get("results", [])
            span_rag.finish(chunks_retrieved=str(len(retrieved_chunks)))
            callback_manager.on_agent_finish(
                self.rag_agent.agent_id,
                self.rag_agent.framework,
                f"Retrieved {len(retrieved_chunks)} relevant code chunks",
            )

            global_blackboard.update_metadata("rag_context", rag_context)
            global_blackboard.update_metadata("retrieved_chunks", retrieved_chunks)

            provenance_tracker.record_step(
                step_id="STEP-0-RAG",
                agent_id=self.rag_agent.agent_id,
                framework=self.rag_agent.framework,
                action_type="rag_retrieval",
                input_payload={"query": raw_prompt, "repo_id": run_sandbox_id},
                output_payload={"chunks_retrieved": len(retrieved_chunks), "context_snippet": rag_context[:200]},
                state_version=global_blackboard.version,
            )
        except Exception as e:
            span_rag.finish(error=str(e))
            callback_manager.on_agent_error(
                self.rag_agent.agent_id,
                self.rag_agent.framework,
                f"RAG indexing/retrieval warning: {str(e)}",
            )

        # -------------------------------------------------------------
        # STEP 3: CrewAI Planner - Requirements Analysis & Plan Breakdown
        # -------------------------------------------------------------
        span_plan = system_tracer.start_span("crewai_planning_phase")
        callback_manager.on_agent_start(self.planner.agent_id, self.planner.framework, "Requirement Decomposition")

        requirement = DeveloperRequirement(
            raw_prompt=raw_prompt,
            target_repo_path=repo_cfg.repo_url or "sample_repo",
            repo_config=repo_cfg,
        )
        plan = self.planner.plan_requirement(requirement, rag_context=rag_context)
        span_plan.finish(tasks_count=str(len(plan.tasks)))
        callback_manager.on_agent_finish(
            self.planner.agent_id,
            self.planner.framework,
            f"Plan generated with {len(plan.tasks)} tasks",
        )

        provenance_tracker.record_step(
            step_id="STEP-1-PLAN",
            agent_id=self.planner.agent_id,
            framework=self.planner.framework,
            action_type="plan",
            input_payload={"prompt": raw_prompt, "has_rag_context": bool(rag_context)},
            output_payload={"plan_id": plan.plan_id, "tasks": [t.id for t in plan.tasks]},
            state_version=global_blackboard.version,
        )

        # -------------------------------------------------------------
        # STEP 4: Google ADK Coder - Code Generation via A2A
        # -------------------------------------------------------------
        span_code = system_tracer.start_span("google_adk_coding_phase")
        callback_manager.on_agent_start(self.coder.agent_id, self.coder.framework, "Sandboxed Code Modification")

        target_file, is_new_file = self._resolve_target_file(
            sandbox_dir=sandbox_dir,
            raw_prompt=raw_prompt,
            retrieved_chunks=retrieved_chunks,
            analysis=global_blackboard.state.analysis,
        )

        persona_guidance = (
            f"\n\n[Engineering Persona Guidance: {self.persona_config.title}]\n"
            f"{self.persona_config.system_instruction_modifier}"
        )
        code_envelope = self.manager.delegate_task_to_adk(
            task_id="TSK-02-CODING",
            role="Coder",
            instruction=f"Implement requirement: {raw_prompt}{persona_guidance}",
            run_id=run_sandbox_id,
            context={
                "sandbox_path": sandbox_path,
                "target_file": target_file,
                "is_new_file": is_new_file,
                "rag_context": rag_context,
                "retrieved_chunks": retrieved_chunks,
                "analysis": global_blackboard.state.analysis.model_dump(mode="json") if global_blackboard.state.analysis else {},
            },
        )
        span_code.finish(status=code_envelope.payload.get("status") if code_envelope else "failed")
        callback_manager.on_agent_finish(
            self.coder.agent_id,
            self.coder.framework,
            f"Code changes posted ({code_envelope.payload.get('patches_count', 0) if code_envelope else 0} patches)",
        )

        provenance_tracker.record_step(
            step_id="STEP-2-CODE",
            agent_id=self.coder.agent_id,
            framework=self.coder.framework,
            action_type="code_generation",
            input_payload={"task_id": "TSK-02-CODING", "instruction": raw_prompt},
            output_payload={"status": "completed", "patches": list(global_blackboard.state.active_patches.keys())},
            state_version=global_blackboard.version,
        )

        # -------------------------------------------------------------
        # STEP 5: Google ADK Tester - Targeted Test Generation & Full-Suite Run via A2A
        # -------------------------------------------------------------
        span_test = system_tracer.start_span("google_adk_test_phase")
        callback_manager.on_agent_start(self.tester.agent_id, self.tester.framework, "Unit Test Synthesis and Execution")

        test_envelope = self.manager.delegate_task_to_adk(
            task_id="TSK-03-TESTING",
            role="TestGenerator",
            instruction="Generate targeted tests for the acceptance criteria and run them with the existing suite",
            run_id=run_sandbox_id,
            context={"sandbox_path": sandbox_path},
        )
        span_test.finish(all_passed=str(test_envelope.payload.get("all_passed") if test_envelope else False))
        callback_manager.on_agent_finish(
            self.tester.agent_id,
            self.tester.framework,
            f"Test results: {test_envelope.payload.get('passed', 0) if test_envelope else 0} passed",
        )

        provenance_tracker.record_step(
            step_id="STEP-3-TEST",
            agent_id=self.tester.agent_id,
            framework=self.tester.framework,
            action_type="test_run",
            input_payload={"sandbox_path": sandbox_path},
            output_payload={"report": test_envelope.payload if test_envelope else {}},
            state_version=global_blackboard.version,
        )

        # -------------------------------------------------------------
        # STEP 6: Failure-Driven Re-planning Loop (Debugger -> Manager -> Planner -> act -> re-test)
        # -------------------------------------------------------------
        span_rec = system_tracer.start_span("failure_driven_replanning_loop")
        recovery_summary = self.recovery_controller.run_recovery_loop(
            run_id=run_sandbox_id,
            sandbox_path=sandbox_path,
            raw_prompt=raw_prompt,
            rag_context=rag_context,
        )
        span_rec.finish(
            cycles=str(recovery_summary["cycles"]),
            stop_reason=str(recovery_summary["stop_reason"])[:80],
        )
        if recovery_summary["cycles"]:
            provenance_tracker.record_step(
                step_id="STEP-3B-RECOVERY",
                agent_id="FailureRecoveryController",
                framework="orchestration",
                action_type="replanning_loop",
                input_payload={"max_cycles": self.recovery_controller.max_recovery_cycles},
                output_payload=recovery_summary,
                state_version=global_blackboard.version,
            )

        # -------------------------------------------------------------
        # STEP 7: Security Audit (AST Safety & Dangerous Syscalls)
        # -------------------------------------------------------------
        span_sec = system_tracer.start_span("security_ast_scan")
        patches_to_scan = [
            (file_path, patch.modified_content)
            for file_path, patch in global_blackboard.state.active_patches.items()
        ]
        audit_report = security_suite.run_full_audit(raw_prompt, patches_to_scan)
        global_blackboard.post_security_audit(audit_report)
        span_sec.finish(passed=str(audit_report.passed))
        callback_manager.on_guardrail_check("ASTSafetyGuardrail", audit_report.passed, audit_report.recommendation)

        provenance_tracker.record_step(
            step_id="STEP-4-SECURITY",
            agent_id="SecurityGuardrailSuite",
            framework="system",
            action_type="security_audit",
            input_payload={"files_scanned": len(patches_to_scan)},
            output_payload={"passed": audit_report.passed, "violations": len(audit_report.ast_security_violations)},
            state_version=global_blackboard.version,
        )

        # -------------------------------------------------------------
        # STEP 8: CrewAI Auditor - Delivery Package Preparation
        # -------------------------------------------------------------
        with usage_ledger.timed_step(self.auditor.agent_id, self.auditor.framework, note="delivery audit"):
            delivery_pkg = self.auditor.audit_and_prepare_delivery()
        if (sandbox_dir / ".git").exists():
            diff_res = mcp_repo_client.get_diff(run_id=run_sandbox_id)
            if diff_res.get("success") and diff_res.get("diff"):
                delivery_pkg.unified_diff = diff_res["diff"]
                global_blackboard.set_final_delivery(delivery_pkg)

        # -------------------------------------------------------------
        # STEP 9: Human-In-The-Loop (HITL) Gatekeeper
        # -------------------------------------------------------------
        hitl_gatekeeper.request_approval()

        if auto_commit_if_approved or settings.AUTO_APPROVE_IN_TESTS:
            hitl_gatekeeper.approve_and_commit(
                run_id=run_sandbox_id,
                approver="Automated Test / CI Suite",
                notes="Auto-approved in isolated sandbox",
                target_repo=repo_cfg.repo_url or "sample_repo",
            )

        # -------------------------------------------------------------
        # STEP 10: Metrics from the usage ledger
        # -------------------------------------------------------------
        reports = global_blackboard.state.test_reports
        final_test_pass = reports[-1].all_passed if reports else False
        regressions = global_blackboard.state.metadata.get("regressions", [])
        success = final_test_pass and audit_report.passed and not regressions
        error = None
        if not success:
            reasons = []
            if not final_test_pass:
                reasons.append(f"tests failing (recovery stop: {recovery_summary['stop_reason']})")
            if regressions:
                reasons.append(f"{len(regressions)} regression(s)")
            if not audit_report.passed:
                reasons.append("AST security audit failed")
            error = "; ".join(reasons)

        coverage = mcp_test_client.run_coverage(sandbox_path)
        quality = {
            "coverage": coverage,
            "retrieved_chunks": [
                {k: c.get(k) for k in ("file_path", "symbol_name", "start_line", "end_line", "score")}
                for c in retrieved_chunks
            ],
        }
        return self._finalize_metrics(
            run_id=run_id,
            overall_start=overall_start,
            raw_prompt=raw_prompt,
            success=success,
            error=error,
            extra_meta={"recovery": recovery_summary, "quality": quality},
            quality=quality,
        )

    # ------------------------------------------------------------------
    def _finalize_metrics(
        self,
        run_id: str,
        overall_start: float,
        raw_prompt: str,
        success: bool,
        error: Optional[str],
        extra_meta: Optional[Dict[str, Any]] = None,
        force_estimated: bool = False,
        quality: Optional[Dict[str, Any]] = None,
    ) -> PerformanceMetrics:
        breakdowns = build_agent_breakdowns(settings.DEFAULT_MODEL)
        total_prompt = sum(b.prompt_tokens for b in breakdowns)
        total_comp = sum(b.completion_tokens for b in breakdowns)

        reports = global_blackboard.state.test_reports
        passed_count = reports[-1].passed if reports else 0
        total_count = reports[-1].total_tests if reports else 0
        pass_rate = (passed_count / total_count) * 100.0 if total_count > 0 else 0.0

        metrics = PerformanceMetrics(
            run_id=run_id,
            architecture_pattern=self.pattern.value,
            persona=self.persona_config.persona_type.value,
            benchmark_task=raw_prompt,
            total_execution_time_seconds=round(time.time() - overall_start, 3),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_comp,
            total_tokens=total_prompt + total_comp,
            total_cost_usd=round(sum(b.estimated_cost_usd for b in breakdowns), 8),
            test_pass_rate_percentage=round(pass_rate, 2),
            tests_passed=passed_count,
            tests_total=total_count,
            failure_recovery_cycles=global_blackboard.state.failure_recovery_cycles,
            tool_call_count=len([e for e in callback_manager.get_stream() if e.get("category") == "tool"]),
            a2a_message_count=len(global_a2a_bus.get_history()),
            agent_breakdowns=breakdowns,
            success=success,
            error_details=error,
            model_name=settings.DEFAULT_MODEL,
            is_measurement_estimated=force_estimated or usage_ledger.any_estimated(),
            quality=quality or {},
        )

        metrics_collector.record_run(
            metrics,
            benchmark_task=raw_prompt,
            persona=self.persona_config.persona_type.value,
            metadata=run_metadata(extra_meta),
        )
        return metrics

    @staticmethod
    def _resolve_target_file(
        sandbox_dir: Path,
        raw_prompt: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        analysis: Optional[Any] = None,
    ) -> Tuple[str, bool]:
        """Resolves target file path dynamically based on prompt, RAG chunks, and sandbox tree.

        Returns:
            Tuple of (relative_file_path, is_new_file)
        """
        import re
        sandbox_dir = sandbox_dir.resolve()

        # 1. Check for explicit file paths mentioned in raw prompt
        prompt_file_matches = re.findall(
            r"[\w\./\-]+\.(?:tsx?|jsx?|py|json|html|css|yaml|yml|md)",
            raw_prompt,
        )
        for cand in prompt_file_matches:
            cand_clean = cand.strip("`'\"()[],;").replace("\\", "/")
            cand_p = (sandbox_dir / cand_clean).resolve()
            if str(cand_p).startswith(str(sandbox_dir)) and cand_p.exists() and cand_p.is_file():
                return str(cand_p.relative_to(sandbox_dir)).replace("\\", "/"), False

        # 2. Check for explicit new file creation in prompt
        new_file_match = re.search(
            r"(?:create|add new file|new file)\s+[`'\"]?([\w\./\-]+\.(?:tsx?|jsx?|py|json|html|css))[`'\"]?",
            raw_prompt,
            re.IGNORECASE,
        )
        if new_file_match:
            new_path = new_file_match.group(1).replace("\\", "/")
            new_p = (sandbox_dir / new_path).resolve()
            if str(new_p).startswith(str(sandbox_dir)):
                return new_path, True

        # 3. Check architectural impact from analysis
        if analysis and getattr(analysis, "architectural_impact", None):
            impact_text = str(analysis.architectural_impact)
            impact_matches = re.findall(
                r"[\w\./\-]+\.(?:tsx?|jsx?|py|json|html|css)",
                impact_text,
            )
            for cand in impact_matches:
                cand_clean = cand.strip("`'\"()[],;").replace("\\", "/")
                cand_p = (sandbox_dir / cand_clean).resolve()
                if str(cand_p).startswith(str(sandbox_dir)) and cand_p.exists() and cand_p.is_file():
                    if "test" not in cand_clean.lower():
                        return str(cand_p.relative_to(sandbox_dir)).replace("\\", "/"), False

        # 4. Check RAG retrieved chunks
        if retrieved_chunks:
            for chunk in retrieved_chunks:
                c_path = chunk.get("file_path")
                if c_path:
                    cand_p = (sandbox_dir / c_path).resolve()
                    if str(cand_p).startswith(str(sandbox_dir)) and cand_p.exists() and cand_p.is_file():
                        if "test" not in c_path.lower() and not c_path.startswith("."):
                            return str(cand_p.relative_to(sandbox_dir)).replace("\\", "/"), False

        # 5. Scan all non-ignored files in sandbox
        ignored_parts = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "build"}
        all_candidates: List[Path] = []
        for p in sandbox_dir.rglob("*"):
            if p.is_file() and not any(part in p.parts for part in ignored_parts):
                if p.suffix.lower() in {".py", ".tsx", ".ts", ".js", ".jsx", ".json", ".html", ".css"}:
                    if "test" not in p.name.lower() and not p.name.startswith("."):
                        all_candidates.append(p)

        if not all_candidates:
            raise RuntimeError(f"No qualifying source files found in sandbox '{sandbox_dir.name}' to implement requirement.")

        # Exclude __init__.py unless it's the only option
        meaningful_candidates = [p for p in all_candidates if p.name != "__init__.py"] or all_candidates

        # 6. Score candidates by keyword matches in prompt
        prompt_words = [w.lower() for w in re.findall(r"\w{3,}", raw_prompt)]
        scored = []
        for p in meaningful_candidates:
            rel = str(p.relative_to(sandbox_dir)).replace("\\", "/").lower()
            stem = p.stem.lower()
            score = 0
            for w in prompt_words:
                if w == stem:
                    score += 10
                elif w in stem or stem in w:
                    score += 3
                elif w in rel:
                    score += 1
            if "src/" in rel:
                score += 1
            scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_path = scored[0][1]
        return str(best_path.relative_to(sandbox_dir)).replace("\\", "/"), False

