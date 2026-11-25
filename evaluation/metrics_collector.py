"""Metrics Collector and Historical Run Aggregator.

Backed by persistent SQLite storage (`evaluation.experiment_db.ExperimentStorage`),
maintaining historical run integrity across process restarts.
"""

from typing import List, Dict, Any, Optional
from config.models_config import calculate_cost
from core.schemas import PerformanceMetrics, AgentTokenUsage
from core.observability import usage_ledger
from evaluation.experiment_db import experiment_storage, ExperimentStorage


def build_agent_breakdowns(model_name: str) -> List[AgentTokenUsage]:
    """Per-agent tokens, cost and latency for the current run, from the usage ledger.

    Shared by all architectures (C1/C2/C3) so they are measured identically."""
    breakdowns = []
    for ag_id, s in usage_ledger.per_agent_summary().items():
        p, c = int(s["prompt_tokens"]), int(s["completion_tokens"])
        breakdowns.append(
            AgentTokenUsage(
                agent_id=ag_id,
                framework=str(s["framework"]),
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=p + c,
                estimated_cost_usd=float(s["cost_usd"]),
                latency_seconds=float(s["latency_seconds"]),
                llm_calls=int(s["llm_calls"]),
                is_estimated=bool(s["is_estimated"]),
            )
        )
    return breakdowns


def llm_breakdowns(model_name: str) -> List[AgentTokenUsage]:
    """Only agents that made LLM calls (excludes pure tool steps such as the baseline test run)."""
    return [b for b in build_agent_breakdowns(model_name) if b.llm_calls > 0]


class MetricsCollector:
    """Stores and aggregates historical execution performance metrics via SQLite."""

    def __init__(self, storage: Optional[ExperimentStorage] = None):
        self.storage = storage or experiment_storage
        self._in_memory_cache: List[PerformanceMetrics] = []

    def record_run(
        self,
        metrics: PerformanceMetrics,
        benchmark_task: Optional[str] = None,
        persona: Optional[str] = None,
        error_details: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records an execution both in-memory and persistently in SQLite.

        `metadata` (per-agent latency/tokens, diagnoses, manager decisions, regressions)
        is stored in the experiments.metadata_json column."""
        self._in_memory_cache.append(metrics)
        if metadata is None:
            metadata = {
                "agent_breakdowns": [b.model_dump(mode="json") for b in metrics.agent_breakdowns],
            }
        self.storage.record_run(
            metrics=metrics,
            benchmark_task=benchmark_task,
            persona=persona,
            execution_status="completed" if metrics.success else "failed",
            error_details=error_details or metrics.error_details,
            metadata=metadata,
        )

    def get_history(self) -> List[PerformanceMetrics]:
        """Returns in-memory cache, reloading from SQLite if memory is empty."""
        if not self._in_memory_cache and self.storage.count() > 0:
            records = self.storage.get_all()
            for r in records:
                try:
                    pm = PerformanceMetrics(
                        run_id=r["run_id"],
                        architecture_pattern=r["architecture_pattern"],
                        persona=r["persona"],
                        benchmark_task=r["benchmark_task"],
                        total_execution_time_seconds=r["execution_time_seconds"],
                        total_prompt_tokens=r["prompt_tokens"],
                        total_completion_tokens=r["completion_tokens"],
                        total_tokens=r["total_tokens"],
                        total_cost_usd=r["estimated_cost_usd"],
                        test_pass_rate_percentage=r["test_pass_rate"],
                        tests_passed=r["tests_passed"],
                        tests_total=r["tests_total"],
                        failure_recovery_cycles=r["failure_recovery_cycles"],
                        tool_call_count=r["tool_call_count"],
                        a2a_message_count=r["a2a_message_count"],
                        success=bool(r["success"]),
                        error_details=r["error_details"],
                        model_name=r["model_name"],
                        is_measurement_estimated=bool(r["is_measurement_estimated"]),
                    )
                    self._in_memory_cache.append(pm)
                except Exception:
                    pass
        return list(self._in_memory_cache)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Returns SQLite aggregate statistics."""
        return self.storage.get_summary_statistics()

    def get_architecture_comparison(self) -> List[Dict[str, Any]]:
        return self.storage.get_architecture_comparison()

    def get_persona_comparison(self) -> List[Dict[str, Any]]:
        return self.storage.get_persona_comparison()

    def clear(self) -> None:
        self._in_memory_cache.clear()
        self.storage.clear()


# Global Singleton
metrics_collector = MetricsCollector()
