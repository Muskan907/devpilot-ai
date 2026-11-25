"""Shared Google ADK invocation helper.

Every Google ADK agent calls Gemini through `ADKInvoker.invoke()`, which:
  * runs the official ADK `Runner` in a fresh session (so each call is independent
    and its token count is not inflated by earlier conversation history),
  * measures wall-clock latency and sums the provider-reported token usage,
  * records the call in the per-agent `usage_ledger`,
  * raises `LLMQuotaExhaustedError` on HTTP 429 / RESOURCE_EXHAUSTED instead of
    silently substituting canned output, so a run that could not reach the model
    is reported as failed rather than as a successful measurement.
"""

import json
import re
import time
import uuid
from typing import Any, Callable, List, Optional, Tuple, Type, TypeVar

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, ValidationError

from core.observability import usage_ledger, LLMQuotaExhaustedError, is_quota_error

T = TypeVar("T", bound=BaseModel)


def resolve_adk_model(model_name: str) -> Any:
    """Returns what LlmAgent(model=...) expects: a Gemini model id as a plain string, or a
    LiteLlm wrapper for non-Gemini providers (e.g. local 'ollama_chat/qwen2.5:1.5b')."""
    if model_name.startswith(("ollama/", "ollama_chat/", "openai/", "anthropic/")):
        from google.adk.models.lite_llm import LiteLlm  # optional dependency (litellm)
        return LiteLlm(model=model_name)
    return model_name


def is_local_model(model_name: str) -> bool:
    return model_name.startswith(("ollama/", "ollama_chat/"))


class ADKInvoker:
    """Runs an ADK LlmAgent through the official Runner and records usage."""

    def __init__(self, agent: LlmAgent, agent_id: str, app_name: str = "multiagent_se"):
        self.agent = agent
        self.agent_id = agent_id
        self.app_name = app_name
        self.user_id = "system_user"
        self.session_service = InMemorySessionService()
        self.runner = Runner(app_name=app_name, agent=agent, session_service=self.session_service)
        self.last_usage: Optional[dict] = None
        m = agent.model
        self.model_name = m if isinstance(m, str) else getattr(m, "model", str(m))

    def invoke(self, prompt: str, max_retries: int = 3) -> str:
        """Sends one prompt; retries transient provider overloads (HTTP 503/UNAVAILABLE)
        with backoff. Quota errors are never retried (they raise LLMQuotaExhaustedError)."""
        for attempt in range(max_retries + 1):
            try:
                return self._invoke_once(prompt)
            except LLMQuotaExhaustedError:
                raise
            except Exception as exc:
                transient = "503" in str(exc) or "UNAVAILABLE" in str(exc) or "overloaded" in str(exc).lower()
                if not transient or attempt == max_retries:
                    raise
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("unreachable")

    def _invoke_once(self, prompt: str) -> str:
        """Sends one prompt to the agent and returns the concatenated text response."""
        session_id = f"{self.agent.name}_{uuid.uuid4().hex[:10]}"
        self.session_service.create_session_sync(
            app_name=self.app_name, user_id=self.user_id, session_id=session_id
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

        text_parts: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        start = time.perf_counter()
        try:
            for event in self.runner.run(user_id=self.user_id, session_id=session_id, new_message=content):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if getattr(part, "text", None) and not getattr(part, "thought", False):
                            text_parts.append(part.text)
                usage = getattr(event, "usage_metadata", None)
                if usage:
                    # Each non-streaming model response carries usage for that one model call;
                    # summing covers multi-step tool-calling turns.
                    prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
                    completion_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
        except Exception as exc:
            latency = time.perf_counter() - start
            usage_ledger.record(
                agent_id=self.agent_id,
                framework="google_adk",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=latency,
                succeeded=False,
                note=f"error: {str(exc)[:120]}",
                model=self.model_name,
            )
            if is_quota_error(exc):
                raise LLMQuotaExhaustedError(f"{self.agent_id}: LLM quota exhausted: {str(exc)[:200]}") from exc
            raise

        latency = time.perf_counter() - start
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_seconds": round(latency, 4),
            # Provider returned no usage data: tokens are unknown, flag the measurement.
            "is_estimated": (prompt_tokens + completion_tokens) == 0,
        }
        usage_ledger.record(
            agent_id=self.agent_id,
            framework="google_adk",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency,
            is_estimated=self.last_usage["is_estimated"],
            model=self.model_name,
        )
        return "".join(text_parts)


def record_simulated_call(
    agent_id: str,
    framework: str,
    prompt_text: str,
    output_text: str,
    latency_seconds: float = 0.0,
) -> dict:
    """Records a deterministic-simulation call. Tokens are approximated (~4 chars/token)
    and the record is explicitly flagged as estimated so it can never be mistaken for a
    real measurement."""
    p = max(1, len(prompt_text) // 4)
    c = max(1, len(output_text) // 4)
    usage_ledger.record(
        agent_id=agent_id,
        framework=framework,
        prompt_tokens=p,
        completion_tokens=c,
        latency_seconds=latency_seconds,
        is_estimated=True,
        note="deterministic_simulation",
    )
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c, "is_estimated": True}


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> Optional[str]:
    """Returns the first fenced code block's body, or None."""
    m = _FENCE_RE.search(text or "")
    return (m.group(1).rstrip() + "\n") if m else None


def parse_json_model(text: str, model: Type[T]) -> T:
    """Parses an LLM response into a Pydantic model, tolerating ```json fences and prose
    around the object. Raises ValueError when no valid object can be extracted."""
    candidates: List[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    first, last = (text or "").find("{"), (text or "").rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first:last + 1])
    errors: List[str] = []
    for cand in candidates:
        try:
            return model.model_validate(json.loads(cand))
        except (json.JSONDecodeError, ValidationError) as e:
            errors.append(str(e)[:200])
    raise ValueError(f"Could not parse {model.__name__} from LLM output: {'; '.join(errors) or 'no JSON object found'}")
