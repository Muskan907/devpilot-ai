"""Multi-Agent Architecture Patterns Specification and Implementations.

Supports comparative evaluation between:
1. Hierarchical-Blackboard-A2A (Proposed Full System)
2. Sequential-Pipeline Pattern (Baseline)
3. Critic-Reflection Loop Pattern
"""

from enum import Enum
from typing import Dict, Any


class AgenticPatternType(str, Enum):
    HIERARCHICAL_BLACKBOARD_A2A = "Hierarchical-Blackboard-A2A"
    SEQUENTIAL_BASELINE = "Sequential-Baseline"
    CRITIC_REFLECTION_LOOP = "Critic-Reflection-Loop"
    HIERARCHICAL_P2P_A2A = "Hierarchical-P2P-A2A"
    HIERARCHICAL_CROSS_MODEL = "Hierarchical-CrossModel"


PATTERN_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    AgenticPatternType.HIERARCHICAL_P2P_A2A.value: {
        "name": "Hierarchical roles with Peer-to-Peer A2A messaging (no Blackboard)",
        "description": (
            "Same agents, tools and model as C1, but agents hand work directly to the next peer via A2A "
            "envelopes that carry all required state; each agent sees only the message it received; "
            "no Manager hub - the Debugger routes failures itself."
        ),
        "strengths": ["No shared-state bottleneck", "Explicit, auditable hand-offs"],
        "weaknesses": ["Larger messages", "No global view for re-planning decisions"],
    },
    AgenticPatternType.HIERARCHICAL_CROSS_MODEL.value: {
        "name": "Hierarchical + Blackboard with a different LLM for Coding and RAG (C4)",
        "description": (
            "Identical to C1 except the Coder runs a local open model through Ollama and the RAG agent "
            "uses learned Ollama embeddings instead of deterministic hashing embeddings."
        ),
        "strengths": ["Zero API cost for coding", "Learned retrieval embeddings"],
        "weaknesses": ["Small local model is weaker at code generation", "Slower on CPU"],
    },
    AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A.value: {
        "name": "Hierarchical Planner with Shared Blackboard and A2A Protocol",
        "description": (
            "CrewAI Planner orchestrates high-level tasks and audits; Google ADK specialized agents "
            "synthesize code and unit tests; communication uses typed A2A envelopes; state is published "
            "to a reactive Shared Blackboard."
        ),
        "strengths": [
            "Framework specialization (CrewAI for management, Google ADK for coding/testing)",
            "Visible decoupled state updates via Blackboard",
            "Cryptographic A2A envelope provenance and non-repudiation",
            "Isolated MCP tool safety",
        ],
        "complexity": "High (Enterprise-grade)",
    },
    AgenticPatternType.SEQUENTIAL_BASELINE.value: {
        "name": "Sequential Single-Agent Baseline",
        "description": (
            "A monolithic single agent executes planning, coding, and testing sequentially in a single context "
            "without inter-framework delegation or shared blackboard."
        ),
        "strengths": ["Low architectural complexity", "Fewer message round-trips"],
        "weaknesses": [
            "Context window exhaustion on large codebases",
            "No separation of concerns between planner, coder, and auditor",
            "Lack of formal inter-agent provenance",
        ],
        "complexity": "Low",
    },
    AgenticPatternType.CRITIC_REFLECTION_LOOP.value: {
        "name": "Generator-Critic Reflection Loop Pattern",
        "description": (
            "Coder synthesizes candidate changes; Critic/Tester reviews and executes tests; feedback "
            "iteratively refines the code before final human gatekeeper approval."
        ),
        "strengths": ["High code accuracy through iterative critique", "Self-healing error recovery"],
        "weaknesses": ["Higher token consumption and latency per run"],
        "complexity": "Medium",
    },
}

