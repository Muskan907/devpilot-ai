"""Human-In-The-Loop (HITL) Gatekeeper Module."""

from datetime import datetime, timezone
from typing import Dict, Any, Optional

from config.settings import settings
from core.blackboard import global_blackboard
from core.observability import callback_manager
from mcp_servers.repo_tool import mcp_repo_client


class HITLGatekeeper:
    """Enforces compulsory human approval before any sandboxed code change is committed to the main codebase."""

    def __init__(self):
        self.require_approval = settings.REQUIRE_HUMAN_APPROVAL

    def evaluate_readiness(self) -> Dict[str, Any]:
        """Assesses whether all prerequisites (tests passing, security audit clean) are satisfied."""
        state = global_blackboard.state
        has_patches = len(state.working_proposals) > 0
        has_reports = len(state.test_reports) > 0
        latest_test_passed = state.test_reports[-1].all_passed if has_reports else False
        security_passed = state.security_audit.passed if state.security_audit else False

        can_approve = has_patches and latest_test_passed and security_passed
        return {
            "can_approve": can_approve,
            "has_patches": has_patches,
            "latest_test_passed": latest_test_passed,
            "security_passed": security_passed,
            "current_status": state.human_approval_status,
        }

    def request_approval(self) -> None:
        """Flags that human intervention is now requested."""
        global_blackboard.set_human_approval("pending", "Awaiting human review")
        callback_manager.on_hitl_event("APPROVAL_REQUESTED", "Awaiting human sign-off on code changes")

    def approve_and_commit(
        self,
        run_id: str,
        approver: str = "Lead Engineer / Faculty Reviewer",
        notes: str = "Approved after reviewing diffs and passing test matrix",
        target_repo: str = "sample_repo",
        user: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Applies human approval and triggers MCP commit to main repository with RBAC validation."""
        if user is not None:
            from core.auth import Permission
            if hasattr(user, "has_permission") and not user.has_permission(Permission.SIGN_OFF_HITL):
                return {
                    "success": False,
                    "error": f"AUTHORIZATION_DENIED: User '{getattr(user, 'user_id', 'unknown')}' lacks permission '{Permission.SIGN_OFF_HITL.value}'.",
                }

        global_blackboard.set_human_approval("approved", f"Approved by {approver}: {notes}")
        callback_manager.on_hitl_event("APPROVAL_GRANTED", f"Signed off by {approver}")

        # Update final delivery package
        state = global_blackboard.state
        if state.final_delivery:
            state.final_delivery.human_approved = True
            state.final_delivery.approver_identity = approver
            state.final_delivery.approval_timestamp = datetime.now(timezone.utc)
            global_blackboard.set_final_delivery(state.final_delivery)

        # Execute safe commit via MCP Repo Tool
        commit_res = mcp_repo_client.commit_to_main(
            run_id=run_id,
            target_repo=target_repo,
            human_approved=True,
            approval_notes=f"{approver}: {notes}",
        )

        return commit_res

    def reject(
        self,
        reason: str = "Code changes did not meet requirements or quality standards",
    ) -> None:
        """Rejects code changes, blocking commit to main."""
        global_blackboard.set_human_approval("rejected", reason)
        callback_manager.on_hitl_event("APPROVAL_REJECTED", reason)


# Global HITL instance
hitl_gatekeeper = HITLGatekeeper()
