"""Failure-Driven Re-planning Loop (synopsis: PLAN -> ACT -> OBSERVE -> REASON -> RE-PLAN -> ACT).

One recovery cycle:
  1. OBSERVE  - the Testing agent's latest report is failing.
  2. REASON   - the Debugging/Review agent (ADK) classifies the failure -> FailureDiagnosis.
  3. DECIDE   - the Manager (CrewAI) chooses: retrieve more context, request a code fix,
                request a test fix, or stop -> ManagerDecision (retry limit enforced).
  4. RE-PLAN  - the Planner (CrewAI) appends debug/verify tasks to the plan.
  5. ACT      - the chosen action is executed (RAG + Coder, Debugger patch, or test repair).
  6. OBSERVE  - tests are re-run; loop until they pass, the Manager stops, or the limit is hit.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from core.blackboard import global_blackboard
from core.observability import callback_manager
from core.schemas import (
    TestResultReport,
    FailureDiagnosis,
    ManagerDecision,
    RecoveryAction,
)
from agents.google_adk import GoogleADKDebuggerAgent
from agents.crewai import CrewAIPlannerAgent


class FailureRecoveryController:
    """Runs the self-healing loop when tests fail."""

    def __init__(
        self,
        max_recovery_cycles: int = 3,
        debugger: Optional[Any] = None,
        planner: Optional[Any] = None,
        manager: Optional[Any] = None,
        coder: Optional[Any] = None,
        tester: Optional[Any] = None,
        rag_agent: Optional[Any] = None,
    ):
        self.max_recovery_cycles = max_recovery_cycles
        self.debugger = debugger or GoogleADKDebuggerAgent()
        self.planner = planner or CrewAIPlannerAgent()
        self._manager = manager
        self._coder = coder
        self._tester = tester
        self._rag_agent = rag_agent

    # Lazily created so lightweight callers (and existing tests) don't build every agent.
    @property
    def manager(self):
        if self._manager is None:
            from agents.crewai import CrewAIManagerAgent
            self._manager = CrewAIManagerAgent()
        return self._manager

    @property
    def coder(self):
        if self._coder is None:
            from agents.google_adk import GoogleADKCoderAgent
            self._coder = GoogleADKCoderAgent()
        return self._coder

    @property
    def tester(self):
        if self._tester is None:
            from agents.google_adk import GoogleADKTestGeneratorAgent
            self._tester = GoogleADKTestGeneratorAgent()
        return self._tester

    @property
    def rag_agent(self):
        if self._rag_agent is None:
            from agents.google_adk import RepositoryRAGAgent
            self._rag_agent = RepositoryRAGAgent()
        return self._rag_agent

    # ------------------------------------------------------------------
    def run_recovery_loop(
        self,
        run_id: str,
        sandbox_path: str,
        raw_prompt: str,
        rag_context: str = "",
    ) -> Dict[str, Any]:
        """Repeats recovery cycles until tests pass, the Manager stops, or the limit is reached."""
        reports = global_blackboard.state.test_reports
        report = reports[-1] if reports else None
        stop_reason = "tests_passed"
        cycles_run = 0

        while report is not None and not report.all_passed:
            outcome = self._run_cycle(run_id, sandbox_path, raw_prompt, report, rag_context)
            if outcome["stopped"]:
                stop_reason = outcome["reason"]
                break
            cycles_run += 1
            rag_context = outcome.get("rag_context", rag_context)
            report = self.tester.run_tests_only(
                run_id=run_id,
                sandbox_path=sandbox_path,
                task_id=f"TSK-RECOVERY-{global_blackboard.state.failure_recovery_cycles}-VERIFY",
            )
        else:
            if report is None:
                stop_reason = "no_test_report"

        return {
            "cycles": global_blackboard.state.failure_recovery_cycles,
            "cycles_run": cycles_run,
            "final_all_passed": bool(report and report.all_passed),
            "stop_reason": stop_reason,
            "decisions": [d.action.value for d in global_blackboard.state.manager_decisions],
            "diagnoses": [d.category.value for d in global_blackboard.state.failure_diagnoses],
        }

    def attempt_recovery(
        self,
        run_id: str,
        sandbox_path: str,
        failed_report: TestResultReport,
        raw_prompt: str = "",
    ) -> Dict[str, Any]:
        """Runs a single recovery cycle (no re-test). Kept for callers that drive the loop themselves."""
        outcome = self._run_cycle(run_id, sandbox_path, raw_prompt, failed_report, "")
        return {
            "recovered": not outcome["stopped"] and outcome.get("applied", False),
            "cycles": global_blackboard.state.failure_recovery_cycles,
            "action": outcome.get("action"),
            "reason": outcome.get("reason"),
            "plan_id": global_blackboard.state.plan.plan_id if global_blackboard.state.plan else None,
        }

    # ------------------------------------------------------------------
    def _run_cycle(
        self,
        run_id: str,
        sandbox_path: str,
        raw_prompt: str,
        report: TestResultReport,
        rag_context: str,
    ) -> Dict[str, Any]:
        next_cycle = global_blackboard.state.failure_recovery_cycles + 1
        callback_manager.on_agent_start(
            self.debugger.agent_id, self.debugger.framework, f"Recovery cycle {next_cycle}: diagnosing failure"
        )

        diagnosis: FailureDiagnosis = self.debugger.diagnose(
            run_id=run_id, failed_report=report, cycle=next_cycle, extra_context=rag_context
        )
        decision: ManagerDecision = self.manager.decide_next_action(
            diagnosis, cycle=next_cycle, max_cycles=self.max_recovery_cycles
        )
        if next_cycle > self.max_recovery_cycles or decision.action == RecoveryAction.STOP:
            callback_manager.on_agent_error(
                "FailureRecoveryController", "orchestration",
                f"Recovery stopped at cycle {next_cycle}: {decision.rationale}",
            )
            return {"stopped": True, "reason": decision.rationale, "action": decision.action.value}

        # RE-PLAN: record the new debug/verify tasks on the shared plan (increments cycle counter).
        self.planner.replan_on_failure(f"{diagnosis.category.value}: {diagnosis.root_cause}")
        task_id = f"TSK-RECOVERY-{global_blackboard.state.failure_recovery_cycles}-DEBUG"

        applied, new_context = self._execute(decision.action, diagnosis, run_id, sandbox_path, raw_prompt, task_id)
        result = {"stopped": False, "applied": applied, "action": decision.action.value, "reason": decision.rationale}
        if new_context is not None:
            result["rag_context"] = new_context
        return result

    def _execute(
        self,
        action: RecoveryAction,
        diagnosis: FailureDiagnosis,
        run_id: str,
        sandbox_path: str,
        raw_prompt: str,
        task_id: str,
    ):
        """Executes the Manager's decision. Returns (applied, new_rag_context_or_None)."""
        feedback = (
            f"\n\nA previous attempt failed tests. Diagnosis ({diagnosis.category.value}): "
            f"{diagnosis.root_cause}\nFailing tests: {', '.join(diagnosis.failing_tests[:10])}\n"
            f"Fix the problem while keeping all existing behaviour."
        )

        if action == RecoveryAction.RETRIEVE_MORE_CONTEXT:
            query = f"{raw_prompt} {diagnosis.root_cause} {' '.join(diagnosis.failing_tests[:5])}"
            rag_res = self.rag_agent.retrieve_context(
                query=query, repo_path=Path(sandbox_path), repo_id=run_id, top_k=10
            )
            new_context = rag_res.get("context", "")
            self._recode(run_id, raw_prompt + feedback, task_id, diagnosis, new_context, rag_res.get("results", []))
            return True, new_context

        if action == RecoveryAction.REQUEST_CODE_FIX:
            if diagnosis.fixed_content:
                return self.debugger.apply_repair(run_id, diagnosis, task_id) is not None, None
            self._recode(run_id, raw_prompt + feedback, task_id, diagnosis,
                         global_blackboard.state.metadata.get("rag_context", ""), [])
            return True, None

        if action == RecoveryAction.REQUEST_TEST_FIX:
            if diagnosis.fixed_content:
                return self.debugger.apply_repair(run_id, diagnosis, task_id) is not None, None
            # No concrete fix: regenerate targeted tests (the next run_tests_only re-verifies).
            self.tester.generate_and_run_tests(run_id=run_id, sandbox_path=sandbox_path, task_id=task_id)
            return True, None

        return False, None

    def _recode(
        self,
        run_id: str,
        instruction: str,
        task_id: str,
        diagnosis: FailureDiagnosis,
        rag_context: str,
        chunks: List[dict],
    ) -> None:
        target = diagnosis.target_file
        if not target or "test" in Path(target).name.lower():
            target = next(
                (p for p in global_blackboard.state.active_patches if "test" not in Path(p).name.lower()),
                None,
            )
        analysis = global_blackboard.state.analysis
        self.coder.implement_changes(
            run_id=run_id,
            task_id=task_id,
            instruction=instruction,
            context={
                "target_file": target,
                "rag_context": rag_context,
                "retrieved_chunks": chunks,
                "analysis": analysis.model_dump(mode="json") if analysis else {},
            },
        )
