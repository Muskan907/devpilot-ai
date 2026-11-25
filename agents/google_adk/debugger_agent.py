"""Google ADK Debugging / Review Agent.

Synopsis role: when tests fail, examine the failure, classify its likely cause
(missing import, wrong context retrieved, incorrect logic, broken test, ...),
recommend the next action to the Manager, and - when it can - propose a corrected
version of the file at fault. The Manager (CrewAI) makes the final decision.
"""

import ast
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from google.adk.agents import LlmAgent

from config.settings import settings
from core.schemas import (
    TaskStatus,
    CodeChangeProposal,
    FilePatch,
    ChangeType,
    TestResultReport,
    FailureDiagnosis,
    FailureCategory,
    RecoveryAction,
)
from core.a2a import A2AEnvelope, A2AMessageType, global_a2a_bus
from core.blackboard import global_blackboard
from core.observability import callback_manager
from mcp_servers.repo_tool import mcp_repo_client
from agents.google_adk.adk_runtime import ADKInvoker, parse_json_model, record_simulated_call

DEBUGGER_INSTRUCTION = (
    "You are an expert debugging and code-review engineer. Given failing test output and the "
    "code involved, you identify the root cause and classify it into exactly one category: "
    "missing_import, wrong_context, incorrect_logic, broken_test, syntax_error, environment, unknown. "
    "Decide whether the production code or the test is wrong. A test is 'broken' only if it "
    "contradicts the requirement or is itself invalid. Recommend one action: request_code_fix, "
    "request_test_fix, retrieve_more_context (the code relies on repository parts you were not "
    "shown), or stop (the failure cannot be fixed by changing code). Be precise and minimal."
)

_PY_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)
_MAX_SNIPPET = 6000


class GoogleADKDebuggerAgent:
    """Google ADK-powered Root-Cause Debugger / Review Agent."""

    def __init__(self, model_name: Optional[str] = None):
        self.agent_id = "GoogleADK:DebuggerAgent"
        self.framework = "google_adk"
        self.model_name = model_name or settings.DEFAULT_MODEL
        self.last_token_usage: Optional[Dict[str, Any]] = None

        self.adk_agent = LlmAgent(
            name="google_adk_debugger",
            description="Autonomous Software Engineering Debugger and Root-Cause Analysis Specialist",
            instruction=DEBUGGER_INSTRUCTION,
            model=self.model_name,
        )
        self.invoker = ADKInvoker(self.adk_agent, self.agent_id)

        global_a2a_bus.register_agent(self.agent_id, self.handle_a2a_message)

    # ------------------------------------------------------------------ A2A
    def handle_a2a_message(self, envelope: A2AEnvelope) -> Optional[A2AEnvelope]:
        if envelope.message_type == A2AMessageType.TASK_DELEGATION:
            task_id = envelope.payload.get("task_id", "TSK-DEBUG")
            run_id = envelope.payload.get("run_id")
            context = envelope.payload.get("context", {})
            reports = global_blackboard.state.test_reports
            if not reports:
                return None
            diagnosis = self.diagnose(
                run_id=run_id,
                failed_report=reports[-1],
                cycle=int(context.get("cycle", global_blackboard.state.failure_recovery_cycles)),
                extra_context=context.get("rag_context", ""),
            )
            return A2AEnvelope(
                sender_id=self.agent_id,
                sender_framework=self.framework,
                recipient_id=envelope.sender_id,
                recipient_framework=envelope.sender_framework,
                correlation_id=envelope.message_id,
                message_type=A2AMessageType.TASK_RESULT,
                payload={
                    "task_id": task_id,
                    "status": "completed",
                    "diagnosis": diagnosis.model_dump(mode="json", exclude={"fixed_content"}),
                    "has_fix": diagnosis.fixed_content is not None,
                    "summary": f"Diagnosed failure as {diagnosis.category.value}",
                },
            )
        return None

    # ------------------------------------------------------------ diagnose
    def diagnose(
        self,
        run_id: str,
        failed_report: TestResultReport,
        cycle: int = 0,
        extra_context: str = "",
    ) -> FailureDiagnosis:
        """Classifies the failure and (optionally) proposes a corrected file. Posts to Blackboard."""
        failing = [
            (item.test_node_id, (item.failure_message or "").strip())
            for item in failed_report.items
            if item.outcome in ("failed", "error")
        ]
        if settings.EXECUTION_MODE == "live":
            diagnosis = self._diagnose_live(run_id, failed_report, failing, extra_context)
        else:
            diagnosis = self._diagnose_rule_based(run_id, failed_report, failing)

        diagnosis.cycle = cycle
        diagnosis.failing_tests = diagnosis.failing_tests or [node for node, _ in failing]
        diagnosis.target_file = self._validate_target(run_id, diagnosis.target_file)
        if diagnosis.target_file is None and diagnosis.recommended_action in (
            RecoveryAction.REQUEST_CODE_FIX, RecoveryAction.REQUEST_TEST_FIX
        ):
            diagnosis.fixed_content = None

        global_blackboard.post_failure_diagnosis(diagnosis)
        callback_manager.on_agent_finish(
            self.agent_id,
            self.framework,
            f"Diagnosis: {diagnosis.category.value} -> {diagnosis.recommended_action.value} ({diagnosis.root_cause[:100]})",
        )
        return diagnosis

    def _diagnose_live(
        self,
        run_id: str,
        report: TestResultReport,
        failing: List[tuple],
        extra_context: str,
    ) -> FailureDiagnosis:
        state = global_blackboard.state
        requirement = state.requirement.raw_prompt if state.requirement else ""
        files = {
            path: self._read(run_id, path)
            for path in state.active_patches.keys()
        }
        files_text = "\n\n".join(
            f"File: {p}\n```python\n{c[:_MAX_SNIPPET]}\n```" for p, c in files.items()
        )
        failures_text = "\n".join(f"- {node}: {msg[:800]}" for node, msg in failing) or "(no per-test details)"
        stdout_tail = (report.raw_stdout or "")[-3000:]
        prompt = (
            f"Requirement: {requirement}\n\n"
            f"Failing tests:\n{failures_text}\n\n"
            f"Pytest output (tail):\n```\n{stdout_tail}\n```\n\n"
            f"Files changed in this run (current content):\n{files_text}\n\n"
            + (f"Additional repository context:\n{extra_context[:_MAX_SNIPPET]}\n\n" if extra_context else "")
            + "Respond with:\n"
            "1. A JSON object in a ```json block with keys: category, root_cause, target_file, "
            "recommended_action, confidence (0-1).\n"
            "2. If recommended_action is request_code_fix or request_test_fix, the COMPLETE corrected "
            "content of target_file in a single ```python block after the JSON.\n"
        )
        callback_manager.on_agent_start(self.agent_id, self.framework, f"Diagnosing {len(failing)} failing test(s)")
        response = self.invoker.invoke(prompt)
        self.last_token_usage = self.invoker.last_usage

        try:
            diagnosis = parse_json_model(response, FailureDiagnosis)
        except ValueError as err:
            callback_manager.on_agent_error(self.agent_id, self.framework, f"Unparseable diagnosis: {err}")
            return FailureDiagnosis(
                category=FailureCategory.UNKNOWN,
                root_cause="Debugger output could not be parsed into a structured diagnosis.",
                recommended_action=RecoveryAction.STOP,
                confidence=0.0,
            )
        code_match = _PY_FENCE_RE.search(response)
        if code_match:
            candidate = code_match.group(1).rstrip() + "\n"
            try:
                ast.parse(candidate)
                diagnosis.fixed_content = candidate
            except SyntaxError:
                diagnosis.fixed_content = None
        return diagnosis

    def _diagnose_rule_based(
        self,
        run_id: str,
        report: TestResultReport,
        failing: List[tuple],
    ) -> FailureDiagnosis:
        """Deterministic offline classifier used in simulation mode and as a teaching baseline."""
        start = time.perf_counter()
        text = " ".join(msg for _, msg in failing) + " " + (report.raw_stdout or "")[-2000:]
        source_target = self._primary_source_file(run_id)
        fixed = None

        if re.search(r"ModuleNotFoundError|ImportError|NameError: name '(math|os|re|json|typing|List|Union|Optional)'", text):
            category, action = FailureCategory.MISSING_IMPORT, RecoveryAction.REQUEST_CODE_FIX
            root = "Code uses a name that is never imported."
            target = source_target
            code = self._read(run_id, target) if target else ""
            if target and target.endswith(".py") and "import math" not in code and "math" in text:
                fixed = "import math\n" + code
        elif "SyntaxError" in text or "IndentationError" in text:
            category, action = FailureCategory.SYNTAX_ERROR, RecoveryAction.REQUEST_CODE_FIX
            root, target = "The modified file is not valid Python.", source_target
        elif re.search(r"fixture '\w+' not found", text):
            category, action = FailureCategory.BROKEN_TEST, RecoveryAction.REQUEST_TEST_FIX
            root, target = "A generated test references a fixture that does not exist.", None
        elif "AttributeError" in text and "has no attribute" in text:
            category, action = FailureCategory.WRONG_CONTEXT, RecoveryAction.RETRIEVE_MORE_CONTEXT
            root = "Code or tests refer to an attribute that does not exist; repository context was likely incomplete."
            target = source_target
        elif failing:
            category, action = FailureCategory.INCORRECT_LOGIC, RecoveryAction.REQUEST_CODE_FIX
            root, target = "Assertions fail: implemented behaviour does not match expectations.", source_target
        else:
            category, action = FailureCategory.ENVIRONMENT, RecoveryAction.STOP
            root, target = "Test run failed without per-test failures (collection or environment problem).", None

        diagnosis = FailureDiagnosis(
            category=category,
            root_cause=root,
            target_file=target,
            recommended_action=action,
            fixed_content=fixed,
            confidence=0.6 if fixed else 0.4,
        )
        self.last_token_usage = record_simulated_call(
            self.agent_id, self.framework, text, root, time.perf_counter() - start
        )
        return diagnosis

    # -------------------------------------------------------------- repair
    def apply_repair(self, run_id: str, diagnosis: FailureDiagnosis, task_id: str) -> Optional[CodeChangeProposal]:
        """Writes the debugger's corrected file into the sandbox. Returns None if there is no fix."""
        if not diagnosis.fixed_content or not diagnosis.target_file:
            return None
        original = self._read(run_id, diagnosis.target_file)
        patch_res = mcp_repo_client.apply_patch(
            run_id=run_id,
            file_path=diagnosis.target_file,
            content=diagnosis.fixed_content,
            justification=f"Debugger repair ({diagnosis.category.value}): {diagnosis.root_cause[:120]}",
        )
        proposal = CodeChangeProposal(
            task_id=task_id,
            author_agent=self.agent_id,
            framework=self.framework,
            patches=[
                FilePatch(
                    file_path=diagnosis.target_file,
                    change_type=ChangeType.MODIFY,
                    original_content=original,
                    modified_content=diagnosis.fixed_content,
                    unified_diff=patch_res.get("diff", ""),
                    affected_symbols=[Path(diagnosis.target_file).stem],
                    justification=f"Debugger repair: {diagnosis.category.value}",
                )
            ],
        )
        global_blackboard.post_code_proposal(proposal)
        global_blackboard.update_task_status(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            output_summary=f"Debugger applied {diagnosis.category.value} fix to {diagnosis.target_file}",
        )
        return proposal

    def diagnose_and_repair(
        self,
        run_id: str,
        task_id: str = "TSK-DEBUG",
        failure_info: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> CodeChangeProposal:
        """Backward-compatible one-shot entry point: diagnose the latest failing report and
        apply the proposed fix. Returns an (possibly empty) proposal."""
        reports = global_blackboard.state.test_reports
        report = reports[-1] if reports else TestResultReport(raw_stdout=failure_info)
        diagnosis = self.diagnose(run_id, report, cycle=global_blackboard.state.failure_recovery_cycles)
        proposal = self.apply_repair(run_id, diagnosis, task_id)
        return proposal or CodeChangeProposal(
            task_id=task_id, author_agent=self.agent_id, framework=self.framework, patches=[]
        )

    def request_replan(self, failure_reason: str) -> None:
        """Sends an A2A envelope requesting the CrewAI Planner to inject a recovery task sequence."""
        envelope = A2AEnvelope(
            sender_id=self.agent_id,
            sender_framework=self.framework,
            recipient_id="CrewAI:PlannerAgent",
            recipient_framework="crewai",
            message_type=A2AMessageType.REPLAN_REQUEST,
            payload={"failure_reason": failure_reason},
        )
        global_a2a_bus.dispatch(envelope)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _read(run_id: str, path: Optional[str]) -> str:
        if not path:
            return ""
        res = mcp_repo_client.read_file(run_id, path)
        return res.get("content", "") if res.get("success") else ""

    @staticmethod
    def _primary_source_file(run_id: str) -> Optional[str]:
        for path in global_blackboard.state.active_patches.keys():
            if "test" not in Path(path).name.lower():
                return path
        for f in mcp_repo_client.list_files(run_id).get("files", []):
            if f.endswith(".py") and "test" not in f.lower() and not f.endswith("__init__.py"):
                return f
        return None

    @staticmethod
    def _validate_target(run_id: str, target: Optional[str]) -> Optional[str]:
        """Only allow repairing files inside the sandbox (no traversal, must exist or be patched)."""
        if not target:
            return None
        norm = target.replace("\\", "/")
        while norm.startswith("./"):
            norm = norm[2:]
        if ".." in norm.split("/") or norm.startswith("/") or ":" in norm:
            return None
        if norm in global_blackboard.state.active_patches:
            return norm
        files = set(mcp_repo_client.list_files(run_id).get("files", []))
        return norm if norm in files else None
