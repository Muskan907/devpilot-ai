"""CrewAI Manager Agent.

Coordinates task execution across agent frameworks using the A2A Protocol,
inspecting blackboard state, and routing tasks to Google ADK workers.
"""

import time
from typing import Dict, Any, Optional
from crewai import Agent, Task, Crew

from config.settings import settings
from config.models_config import get_llm_for_crewai
from core.schemas import (
    TaskStatus,
    TaskFramework,
    FailureDiagnosis,
    ManagerDecision,
    RecoveryAction,
)
from core.a2a import A2AEnvelope, A2AMessageType, global_a2a_bus
from core.blackboard import global_blackboard
from core.observability import (
    callback_manager,
    usage_ledger,
    LLMQuotaExhaustedError,
    is_quota_error,
)


class CrewAIManagerAgent:
    """CrewAI-powered Manager / Orchestration Agent."""

    def __init__(self, model_name: Optional[str] = None):
        self.agent_id = "CrewAI:ManagerAgent"
        self.framework = "crewai"
        self.model_name = model_name or settings.DEFAULT_MODEL
        self.llm = get_llm_for_crewai(self.model_name)

        self.crewai_agent = Agent(
            role="Engineering Manager & Multi-Agent Dispatcher",
            goal="Coordinate specialized agent execution, monitor blackboard state, and enforce protocol compliance.",
            backstory=(
                "You are an Engineering Manager who oversees autonomous software engineering teams. "
                "You dispatch work to Google ADK specialist agents via the A2A protocol and ensure "
                "continuous progress on the shared blackboard."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )

        global_a2a_bus.register_agent(self.agent_id, self.handle_a2a_message)

    def handle_a2a_message(self, envelope: A2AEnvelope) -> Optional[A2AEnvelope]:
        """Handles incoming A2A messages from Google ADK workers."""
        if envelope.message_type == A2AMessageType.TASK_RESULT:
            task_id = envelope.payload.get("task_id")
            status = envelope.payload.get("status", "completed")
            summary = envelope.payload.get("summary", "Task executed successfully")
            if task_id:
                global_blackboard.update_task_status(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED if status == "completed" else TaskStatus.FAILED,
                    output_summary=summary,
                )
        return None

    def delegate_task_to_adk(
        self,
        task_id: str,
        role: str,
        instruction: str,
        run_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[A2AEnvelope]:
        """Dispatches an A2A task delegation envelope to Google ADK agent."""
        recipient_id = f"GoogleADK:{role}Agent"
        envelope = A2AEnvelope(
            sender_id=self.agent_id,
            sender_framework=self.framework,
            recipient_id=recipient_id,
            recipient_framework="google_adk",
            message_type=A2AMessageType.TASK_DELEGATION,
            payload={
                "task_id": task_id,
                "role": role,
                "instruction": instruction,
                "run_id": run_id,
                "context": context or {},
            },
        )
        global_blackboard.update_task_status(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            output_summary=f"Delegated to {recipient_id} via A2A protocol",
        )
        return global_a2a_bus.dispatch(envelope)

    # ------------------------------------------------------------------
    # Failure-driven re-planning (synopsis: REASON -> RE-PLAN step)
    # ------------------------------------------------------------------
    def decide_next_action(
        self,
        diagnosis: FailureDiagnosis,
        cycle: int,
        max_cycles: int,
    ) -> ManagerDecision:
        """Decides what happens after a failed test run, then enforces hard policy rules.

        Live mode: a real CrewAI/Gemini call reasons over the diagnosis and the run history.
        Simulation mode: follows the Debugger's recommendation (deterministic baseline).
        Hard rules (always applied): the retry limit, and stopping when the same failure
        category repeats with no progress.
        """
        history = [
            f"cycle {d.cycle}: {d.category.value} -> manager chose "
            f"{next((m.action.value for m in global_blackboard.state.manager_decisions if m.cycle == d.cycle), 'n/a')}"
            for d in global_blackboard.state.failure_diagnoses[:-1]
        ]

        if settings.EXECUTION_MODE == "live":
            decision = self._decide_live(diagnosis, cycle, max_cycles, history)
        else:
            decision = ManagerDecision(
                action=diagnosis.recommended_action,
                rationale=f"[simulation] Following debugger recommendation for {diagnosis.category.value}.",
                cycle=cycle,
            )
            usage_ledger.record(
                agent_id=self.agent_id, framework=self.framework,
                prompt_tokens=60, completion_tokens=20, is_estimated=True, note="deterministic_simulation",
            )

        decision.cycle = cycle
        decision = self._apply_policy(decision, diagnosis, cycle, max_cycles)
        global_blackboard.post_manager_decision(decision)
        callback_manager.on_agent_finish(
            self.agent_id, self.framework,
            f"Re-plan decision (cycle {cycle}): {decision.action.value} - {decision.rationale[:100]}",
        )
        return decision

    def _decide_live(self, diagnosis: FailureDiagnosis, cycle: int, max_cycles: int, history: list) -> ManagerDecision:
        requirement = global_blackboard.state.requirement.raw_prompt if global_blackboard.state.requirement else ""
        task = Task(
            description=(
                "A test run failed during an autonomous code change. Decide the next step.\n"
                f"Requirement: {requirement}\n"
                f"Recovery cycle: {cycle} of max {max_cycles}\n"
                f"Debugger diagnosis: category={diagnosis.category.value}; "
                f"root_cause={diagnosis.root_cause}; failing_tests={diagnosis.failing_tests[:10]}; "
                f"debugger_recommends={diagnosis.recommended_action.value}; "
                f"has_ready_fix={'yes' if diagnosis.fixed_content else 'no'}; "
                f"confidence={diagnosis.confidence}\n"
                f"Previous cycles: {history or 'none'}\n\n"
                "Choose exactly one action:\n"
                "- retrieve_more_context: the failure suggests the coder lacked repository knowledge "
                "(wrong names, unknown APIs, missing dependencies); fetch more context and re-implement.\n"
                "- request_code_fix: production code is wrong and can be corrected.\n"
                "- request_test_fix: the generated test contradicts the requirement or is invalid.\n"
                "- stop: further attempts are unlikely to help (environment problem, repeated identical "
                "failures, or requirement impossible).\n"
                "Give a one-sentence rationale."
            ),
            expected_output="A ManagerDecision with fields action and rationale.",
            agent=self.crewai_agent,
            output_pydantic=ManagerDecision,
        )
        start = time.perf_counter()
        try:
            crew_res = Crew(agents=[self.crewai_agent], tasks=[task], verbose=False).kickoff()
        except Exception as exc:
            usage_ledger.record(
                agent_id=self.agent_id, framework=self.framework,
                latency_seconds=time.perf_counter() - start, succeeded=False, note=str(exc)[:120],
            )
            if is_quota_error(exc):
                raise LLMQuotaExhaustedError(f"{self.agent_id}: LLM quota exhausted: {str(exc)[:200]}") from exc
            raise
        latency = time.perf_counter() - start
        tu = getattr(crew_res, "token_usage", None)
        p = int(getattr(tu, "prompt_tokens", 0) or 0)
        c = int(getattr(tu, "completion_tokens", 0) or 0)
        usage_ledger.record(
            agent_id=self.agent_id, framework=self.framework,
            prompt_tokens=p, completion_tokens=c, latency_seconds=latency, is_estimated=(p + c) == 0,
        )

        decision = task.output.pydantic if task.output else None
        if not isinstance(decision, ManagerDecision):
            return ManagerDecision(
                action=diagnosis.recommended_action,
                rationale="Manager output was not structured; defaulting to the debugger's recommendation.",
                cycle=cycle,
                overridden_by_policy=True,
            )
        return decision

    @staticmethod
    def _apply_policy(
        decision: ManagerDecision,
        diagnosis: FailureDiagnosis,
        cycle: int,
        max_cycles: int,
    ) -> ManagerDecision:
        """Hard stopping conditions that the LLM is not allowed to override."""
        if cycle > max_cycles:
            return decision.model_copy(update={
                "action": RecoveryAction.STOP,
                "rationale": f"Retry limit reached ({max_cycles} cycles). Original choice: {decision.action.value}.",
                "overridden_by_policy": True,
            })
        previous = [d for d in global_blackboard.state.failure_diagnoses if d.cycle < cycle]
        if (
            len(previous) >= 2
            and all(d.category == diagnosis.category for d in previous[-2:])
            and all(set(d.failing_tests) == set(diagnosis.failing_tests) for d in previous[-2:])
            and decision.action != RecoveryAction.STOP
        ):
            return decision.model_copy(update={
                "action": RecoveryAction.STOP,
                "rationale": "Same failure repeated for three cycles with no progress; stopping.",
                "overridden_by_policy": True,
            })
        return decision
