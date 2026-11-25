"""Per-Agent Usage Ledger.

Single source of truth for how many tokens each agent consumed, how long each
agent took, and whether those numbers are real (returned by the LLM provider)
or estimated (deterministic simulation mode / provider gave no usage data).

Every LLM call in the system records itself here. Orchestrators build the
per-agent breakdown of a run from this ledger instead of hardcoded constants.
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional


class LLMQuotaExhaustedError(RuntimeError):
    """Raised when the LLM provider rejects a call for quota / rate-limit reasons.

    Runs that hit this are recorded as failed; they are never silently completed
    with canned output."""


def is_quota_error(exc: BaseException) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "quota" in text.lower()


@dataclass
class AgentCallRecord:
    """One measured call made by an agent (LLM call or non-LLM work step)."""
    agent_id: str
    framework: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    is_estimated: bool = False
    is_llm_call: bool = True
    succeeded: bool = True
    note: str = ""
    model: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def cost_usd(self) -> float:
        """Cost from the pricing table for the model actually used; local models are free."""
        if not self.is_llm_call or self.model.startswith(("ollama/", "ollama_chat/")):
            return 0.0
        from config.models_config import calculate_cost  # local import: avoid import cycle
        from config.settings import settings
        return calculate_cost(self.model or settings.DEFAULT_MODEL, self.prompt_tokens, self.completion_tokens)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class UsageLedger:
    """Thread-safe ledger of per-agent calls for the current run."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: List[AgentCallRecord] = []

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def record(
        self,
        agent_id: str,
        framework: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_seconds: float = 0.0,
        is_estimated: bool = False,
        is_llm_call: bool = True,
        succeeded: bool = True,
        note: str = "",
        model: str = "",
    ) -> AgentCallRecord:
        rec = AgentCallRecord(
            agent_id=agent_id,
            framework=framework,
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            latency_seconds=round(float(latency_seconds or 0.0), 4),
            is_estimated=bool(is_estimated),
            is_llm_call=bool(is_llm_call),
            succeeded=bool(succeeded),
            note=note,
            model=model,
        )
        with self._lock:
            self._records.append(rec)
        return rec

    @contextmanager
    def timed_step(self, agent_id: str, framework: str, note: str = "") -> Iterator[None]:
        """Measures latency of a non-LLM agent step (e.g. RAG retrieval, auditing)."""
        start = time.perf_counter()
        ok = True
        try:
            yield
        except Exception:
            ok = False
            raise
        finally:
            self.record(
                agent_id=agent_id,
                framework=framework,
                latency_seconds=time.perf_counter() - start,
                is_llm_call=False,
                succeeded=ok,
                note=note,
            )

    def records(self) -> List[AgentCallRecord]:
        with self._lock:
            return list(self._records)

    def per_agent_summary(self) -> Dict[str, Dict[str, object]]:
        """Aggregates calls per agent: tokens, latency, call count, estimated flag."""
        summary: Dict[str, Dict[str, object]] = {}
        with self._lock:
            for r in self._records:
                s = summary.setdefault(r.agent_id, {
                    "agent_id": r.agent_id,
                    "framework": r.framework,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_seconds": 0.0,
                    "llm_calls": 0,
                    "steps": 0,
                    "is_estimated": False,
                    "failed_calls": 0,
                    "cost_usd": 0.0,
                    "models": [],
                })
                s["prompt_tokens"] += r.prompt_tokens
                s["completion_tokens"] += r.completion_tokens
                s["total_tokens"] += r.total_tokens
                s["latency_seconds"] = round(float(s["latency_seconds"]) + r.latency_seconds, 4)
                s["steps"] += 1
                if r.is_llm_call:
                    s["llm_calls"] += 1
                if r.is_estimated:
                    s["is_estimated"] = True
                if not r.succeeded:
                    s["failed_calls"] += 1
                s["cost_usd"] = round(float(s["cost_usd"]) + r.cost_usd, 8)
                if r.is_llm_call and r.model and r.model not in s["models"]:
                    s["models"].append(r.model)
        return summary

    def any_estimated(self) -> bool:
        with self._lock:
            return any(r.is_estimated for r in self._records if r.is_llm_call)

    def to_json(self) -> List[Dict[str, object]]:
        with self._lock:
            return [{**asdict(r), "cost_usd": r.cost_usd} for r in self._records]


usage_ledger = UsageLedger()
