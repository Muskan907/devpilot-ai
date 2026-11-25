"""C3-P2P: Hierarchical agent roles with PEER-TO-PEER messaging instead of the Blackboard.

Synopsis configuration C3 ("hierarchical architecture with peer-to-peer messaging instead of
Blackboard"). Same agents, tools and model as C1, but:

  * No shared Blackboard for coordination. Every hand-off is an A2A envelope sent directly
    from one peer to the next, and the envelope carries ALL the state the recipient needs
    (requirement, plan, patches, test report, diagnosis).
  * Message-local views. Before an agent acts, its working view is rebuilt ONLY from the
    payload of the message it received, so no agent can read state it was not sent.
  * No Manager hub. On failure the Debugger decides the next hop itself
    (RAG for more context, Coder for a fix, Test agent for a test fix, or stop).

Flow: RAG -> Planner -> Coder -> Tester -> (fail) Debugger -> {RAG -> Coder | Coder} -> Tester ...
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

from config.settings import settings
from core.a2a import A2AEnvelope, A2AMessageType, global_a2a_bus
from core.blackboard import global_blackboard
from core.guardrails import security_suite
from core.hitl import hitl_gatekeeper
from core.observability import callback_manager, usage_ledger
from core.schemas import (
    CodeChangeProposal,
    DeveloperRequirement,
    FilePatch,
    PerformanceMetrics,
    RecoveryAction,
    RepositoryConfig,
    RequirementAnalysis,
    TestResultReport,
)
from mcp_servers.repo_tool import mcp_repo_client
from mcp_servers.test_runner import mcp_test_client
from agents.crewai import CrewAIPlannerAgent
from agents.google_adk import (
    GoogleADKCoderAgent,
    GoogleADKDebuggerAgent,
    GoogleADKTestGeneratorAgent,
    RepositoryRAGAgent,
)
from evaluation.metrics_collector import llm_breakdowns
from orchestration.patterns import AgenticPatternType
from orchestration.personas import PersonaConfig, PersonaType, get_persona

PEER_FRAMEWORK = {
    "RAG": "google_adk", "Planner": "crewai", "Coder": "google_adk",
    "Tester": "google_adk", "Debugger": "google_adk",
}


class PeerToPeerOrchestrator:
    """Decentralised peer-to-peer message passing between the same specialist agents as C1."""

    def __init__(
        self,
        persona: Union[str, PersonaType, PersonaConfig] = PersonaType.BALANCED_ENGINEER,
        max_recovery_cycles: int = 3,
    ):
        self.pattern = AgenticPatternType.HIERARCHICAL_P2P_A2A
        self.persona_config = persona if isinstance(persona, PersonaConfig) else get_persona(persona)
        self.max_recovery_cycles = max_recovery_cycles
        self.rag = RepositoryRAGAgent()
        self.planner = CrewAIPlannerAgent()
        self.coder = GoogleADKCoderAgent()
        self.tester = GoogleADKTestGeneratorAgent()
        self.debugger = GoogleADKDebuggerAgent()

    # ------------------------------------------------------------ messaging
    def _send(self, sender: str, recipient: str, payload: Dict[str, Any], kind=A2AMessageType.TASK_DELEGATION) -> Dict[str, Any]:
        """Direct peer-to-peer hand-off. The envelope (signed, logged on the bus) carries the full payload."""
        envelope = A2AEnvelope(
            sender_id=f"P2P:{sender}",
            sender_framework=PEER_FRAMEWORK[sender],
            recipient_id=f"P2P:{recipient}",
            recipient_framework=PEER_FRAMEWORK[recipient],
            message_type=kind,
            payload=payload,
        )
        global_a2a_bus.dispatch(envelope)  # logged + integrity-checked; delivery handled below
        callback_manager.on_agent_start(f"P2P:{recipient}", PEER_FRAMEWORK[recipient], f"Message from {sender}")
        return envelope.payload

    @staticmethod
    def _materialize(msg: Dict[str, Any]) -> None:
        """Rebuilds the recipient's working view from the message payload ONLY."""
        global_blackboard.reset()
        if msg.get("requirement"):
            global_blackboard.post_requirement(DeveloperRequirement.model_validate(msg["requirement"]))
        if msg.get("analysis"):
            global_blackboard.post_analysis(RequirementAnalysis.model_validate(msg["analysis"]))
        if msg.get("patches"):
            global_blackboard.post_code_proposal(CodeChangeProposal(
                task_id="P2P-VIEW", author_agent="p2p-message", framework="p2p",
                patches=[FilePatch.model_validate(p) for p in msg["patches"].values()],
            ))
        if msg.get("baseline"):
            global_blackboard.update_metadata("baseline_test_report", msg["baseline"])
        if msg.get("report"):
            global_blackboard.post_test_report(TestResultReport.model_validate(msg["report"]))

    @staticmethod
    def _patches() -> Dict[str, Any]:
        return {p: fp.model_dump(mode="json") for p, fp in global_blackboard.state.active_patches.items()}

    # ------------------------------------------------------------------
    def execute_requirement(
        self,
        raw_prompt: str,
        target_repo: Union[str, RepositoryConfig] = "sample_repo",
        auto_commit_if_approved: bool = False,
    ) -> PerformanceMetrics:
        run_id = f"RUN-P2P-{uuid.uuid4().hex[:8]}"
        start = time.time()
        global_blackboard.reset()
        global_a2a_bus.clear()
        usage_ledger.reset()

        ok, violations = security_suite.prompt_guard.validate_prompt(raw_prompt)
        if not ok:
            return PerformanceMetrics(run_id=run_id, architecture_pattern=self.pattern.value,
                                      persona=self.persona_config.persona_type.value, benchmark_task=raw_prompt,
                                      success=False, error_details=f"Prompt rejected: {violations[0].details}")

        repo_cfg = target_repo if isinstance(target_repo, RepositoryConfig) else RepositoryConfig(
            repo_url=target_repo, base_branch="main")
        sb = (mcp_repo_client.clone_repository(repo_url=repo_cfg.repo_url, base_branch=repo_cfg.base_branch)
              if repo_cfg.is_github else mcp_repo_client.create_sandbox(repo_cfg.repo_url))
        if not sb.get("success"):
            return PerformanceMetrics(run_id=run_id, architecture_pattern=self.pattern.value, success=False,
                                      error_details=f"Sandbox creation failed: {sb.get('error')}")
        sid, spath = sb["run_id"], sb["sandbox_path"]
        sdir = Path(spath)
        baseline = mcp_test_client.run_tests(spath)
        baseline_report = baseline.get("report") if baseline.get("success") else None

        requirement = DeveloperRequirement(raw_prompt=raw_prompt, target_repo_path=repo_cfg.repo_url or "sample_repo",
                                           repo_config=repo_cfg)

        # RAG peer -> Planner peer
        with usage_ledger.timed_step(self.rag.agent_id, self.rag.framework, note="index + retrieve"):
            rag_res = self.rag.retrieve_context(query=raw_prompt, repo_path=sdir, repo_id=sid, top_k=5)
        msg = self._send("RAG", "Planner", {
            "requirement": requirement.model_dump(mode="json"),
            "rag_context": rag_res.get("context", ""), "chunks": rag_res.get("results", []),
        })
        retrieved_chunks = msg["chunks"]

        # Planner peer -> Coder peer
        self._materialize(msg)
        self.planner.plan_requirement(requirement, rag_context=msg["rag_context"])
        analysis = global_blackboard.state.analysis
        from orchestration.engine import AutonomousEngineeringOrchestrator  # local: avoid import cycle
        target_file, is_new = AutonomousEngineeringOrchestrator._resolve_target_file(
            sandbox_dir=sdir, raw_prompt=raw_prompt, retrieved_chunks=retrieved_chunks, analysis=analysis)
        msg = self._send("Planner", "Coder", {
            "requirement": msg["requirement"], "analysis": analysis.model_dump(mode="json") if analysis else None,
            "rag_context": msg["rag_context"], "target_file": target_file, "is_new_file": is_new,
        })

        # Coder peer -> Tester peer
        msg = self._code(msg, sid, raw_prompt, "P2P-CODE-1")
        msg = self._send("Coder", "Tester", {**msg, "baseline": baseline_report})
        self._materialize(msg)
        report = self.tester.generate_and_run_tests(run_id=sid, sandbox_path=spath, task_id="P2P-TEST-1")
        msg = {**msg, "patches": self._patches(), "report": report.model_dump(mode="json")}

        # Failure handling: Debugger decides the next hop itself (no Manager)
        cycles, stop_reason = 0, "tests_passed"
        while not report.all_passed:
            if cycles >= self.max_recovery_cycles:
                stop_reason = f"retry limit ({self.max_recovery_cycles})"
                break
            cycles += 1
            msg = self._send("Tester", "Debugger", {**msg, "cycle": cycles})
            self._materialize(msg)
            diagnosis = self.debugger.diagnose(run_id=sid, failed_report=report, cycle=cycles,
                                               extra_context=msg.get("rag_context", ""))
            action = diagnosis.recommended_action
            feedback = (f"\n\nPrevious attempt failed. Diagnosis ({diagnosis.category.value}): {diagnosis.root_cause}\n"
                        f"Failing tests: {', '.join(diagnosis.failing_tests[:10])}")
            if action == RecoveryAction.STOP:
                stop_reason = f"debugger stopped: {diagnosis.root_cause[:80]}"
                break
            if action == RecoveryAction.RETRIEVE_MORE_CONTEXT:
                q = f"{raw_prompt} {diagnosis.root_cause}"
                self._send("Debugger", "RAG", {"query": q}, A2AMessageType.REPLAN_REQUEST)
                with usage_ledger.timed_step(self.rag.agent_id, self.rag.framework, note="re-retrieve"):
                    more = self.rag.retrieve_context(query=q, repo_path=sdir, repo_id=sid, top_k=10)
                msg = self._send("RAG", "Coder", {**msg, "rag_context": more.get("context", "")})
                msg = self._code(msg, sid, raw_prompt + feedback, f"P2P-CODE-{cycles + 1}")
            elif diagnosis.fixed_content:
                self.debugger.apply_repair(sid, diagnosis, f"P2P-FIX-{cycles}")
                msg = {**msg, "patches": self._patches()}
            elif action == RecoveryAction.REQUEST_TEST_FIX:
                msg = self._send("Debugger", "Tester", msg)
                self._materialize(msg)
                self.tester.generate_and_run_tests(run_id=sid, sandbox_path=spath, task_id=f"P2P-RETEST-{cycles}")
                msg = {**msg, "patches": self._patches()}
            else:
                msg = self._send("Debugger", "Coder", msg)
                msg = self._code(msg, sid, raw_prompt + feedback, f"P2P-CODE-{cycles + 1}")
            msg = self._send("Coder" if action != RecoveryAction.REQUEST_TEST_FIX else "Debugger", "Tester", msg)
            self._materialize(msg)
            report = self.tester.run_tests_only(run_id=sid, sandbox_path=spath, task_id=f"P2P-VERIFY-{cycles}")
            msg = {**msg, "patches": self._patches(), "report": report.model_dump(mode="json")}

        regressions = global_blackboard.state.metadata.get("regressions", [])
        audit = security_suite.run_full_audit(
            raw_prompt, [(p, fp["modified_content"]) for p, fp in (msg.get("patches") or {}).items()])
        hitl_gatekeeper.request_approval()

        breakdowns = llm_breakdowns(settings.DEFAULT_MODEL)
        p = sum(b.prompt_tokens for b in breakdowns)
        c = sum(b.completion_tokens for b in breakdowns)
        success = report.all_passed and audit.passed and not regressions
        return PerformanceMetrics(
            run_id=run_id,
            architecture_pattern=self.pattern.value,
            persona=self.persona_config.persona_type.value,
            benchmark_task=raw_prompt,
            total_execution_time_seconds=round(time.time() - start, 3),
            total_prompt_tokens=p,
            total_completion_tokens=c,
            total_tokens=p + c,
            total_cost_usd=round(sum(b.estimated_cost_usd for b in breakdowns), 8),
            test_pass_rate_percentage=round(100.0 * report.passed / report.total_tests, 2) if report.total_tests else 0.0,
            tests_passed=report.passed,
            tests_total=report.total_tests,
            failure_recovery_cycles=cycles,
            tool_call_count=len([e for e in callback_manager.get_stream() if e.get("category") == "tool"]),
            a2a_message_count=len(global_a2a_bus.get_history()),
            agent_breakdowns=breakdowns,
            success=success,
            error_details=None if success else (
                f"P2P run ended ({stop_reason}): tests_all_passed={report.all_passed}, "
                f"regressions={len(regressions)}, audit_passed={audit.passed}"),
            model_name=settings.DEFAULT_MODEL,
            is_measurement_estimated=usage_ledger.any_estimated(),
            quality={
                "coverage": mcp_test_client.run_coverage(spath),
                "retrieved_chunks": [
                    {k: c.get(k) for k in ("file_path", "symbol_name", "start_line", "end_line", "score")}
                    for c in retrieved_chunks
                ],
            },
        )

    def _code(self, msg: Dict[str, Any], sid: str, instruction: str, task_id: str) -> Dict[str, Any]:
        """Coder peer acts on its message-local view and returns the updated payload."""
        self._materialize(msg)
        persona = f"\n\n[Engineering Persona Guidance: {self.persona_config.title}]\n{self.persona_config.system_instruction_modifier}"
        target = msg.get("target_file") or next(
            (p for p in (msg.get("patches") or {}) if "test" not in Path(p).name.lower()), None)
        self.coder.implement_changes(
            run_id=sid, task_id=task_id, instruction=f"Implement requirement: {instruction}{persona}",
            context={"target_file": target, "is_new_file": bool(msg.get("is_new_file")) and task_id.endswith("-1"),
                     "rag_context": msg.get("rag_context", ""), "analysis": msg.get("analysis") or {}},
        )
        # Keep previously-sent patches (e.g. generated tests) and overlay the coder's new ones.
        return {**msg, "patches": {**(msg.get("patches") or {}), **self._patches()}}
