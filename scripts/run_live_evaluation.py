"""Live evaluation campaign (synopsis Section 2.3): 10 tasks x 4 configurations = 40 runs.

  C1  Hierarchical + Blackboard + A2A               (primary design)
  C2  Sequential pipeline, same agents and tools    (does a manager-led design help?)
  C3  Hierarchical roles, peer-to-peer messaging    (Blackboard vs P2P messaging)
  C4  C1 with a different Coder LLM + RAG embedding (which LLM suits which role?)
  CR  Critic-reflection loop                        (extra agentic pattern, optional)

Per run it records: success, tests, regressions, repair cycles, end-to-end and per-agent
latency, tokens, cost, coverage (MCP tool), retrieval quality vs. ground truth, and an
LLM-as-Judge rubric score from a different model than the generator.

Usage (project root):
    .venv\\Scripts\\python.exe scripts\\run_live_evaluation.py                 # all 40, resumes
    .venv\\Scripts\\python.exe scripts\\run_live_evaluation.py --archs C1 C4 --tasks T1 T2
    .venv\\Scripts\\python.exe scripts\\run_live_evaluation.py --fresh         # start over
A run that hits quota or a provider error is recorded as failed/excluded - never as a success.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["EXECUTION_MODE"] = "live"

from config.settings import settings  # noqa: E402
from core.blackboard import global_blackboard  # noqa: E402
from evaluation.retrieval_metrics import retrieval_quality  # noqa: E402
from orchestration.engine import AutonomousEngineeringOrchestrator  # noqa: E402
from orchestration.patterns import AgenticPatternType  # noqa: E402
from orchestration.personas import PersonaType  # noqa: E402

CALC = "src/calculator.py"
TASKS = {
    "T1": ("Add a standard_deviation(numbers, sample=False) method to AdvancedCalculator in src/calculator.py. "
           "It must raise ValueError for an empty list, and for sample=True with fewer than two values.",
           ["mean", "AdvancedCalculator"]),
    "T2": ("Add a median(numbers) method to AdvancedCalculator in src/calculator.py that returns the middle value "
           "(average of the two middle values for even-length input) and raises ValueError for an empty list.",
           ["mean", "AdvancedCalculator"]),
    "T3": ("Add a safe_percentage(part, whole) method to AdvancedCalculator in src/calculator.py that returns "
           "part / whole * 100 rounded to 2 decimals and raises ValueError when whole is 0.",
           ["divide", "AdvancedCalculator"]),
    "T4": ("Add a variance(numbers, sample=False) method to AdvancedCalculator in src/calculator.py returning the "
           "population variance (or sample variance when sample=True); raise ValueError for an empty list.",
           ["mean", "AdvancedCalculator"]),
    "T5": ("Add a factorial(n) method to AdvancedCalculator in src/calculator.py that returns n! for non-negative "
           "integers and raises ValueError for negative numbers or non-integers.",
           ["power", "AdvancedCalculator"]),
    "T6": ("Add a modulo(a, b) method to AdvancedCalculator in src/calculator.py that returns a % b and raises "
           "ZeroDivisionError with the message 'Cannot take modulo by zero' when b is 0.",
           ["divide", "AdvancedCalculator"]),
    "T7": ("Add a compound_interest(principal, rate, years) method to AdvancedCalculator in src/calculator.py "
           "returning principal * (1 + rate) ** years rounded to 2 decimals; raise ValueError if principal or "
           "years is negative.",
           ["power", "AdvancedCalculator"]),
    "T8": ("Add a mode(numbers) method to AdvancedCalculator in src/calculator.py returning the most frequent "
           "value (the smallest one on ties) and raising ValueError for an empty list.",
           ["mean", "AdvancedCalculator"]),
    "T9": ("Add an nth_root(x, n) method to AdvancedCalculator in src/calculator.py returning the real n-th root "
           "of x; raise ValueError when n is 0 or when n is even and x is negative.",
           ["power", "AdvancedCalculator"]),
    "T10": ("Add a clamp(value, low, high) method to AdvancedCalculator in src/calculator.py that limits value "
            "to the range [low, high] and raises ValueError when low is greater than high.",
            ["AdvancedCalculator"]),
}

ARCHS = {
    "C1": AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A,
    "C2": AgenticPatternType.SEQUENTIAL_BASELINE,
    "C3": AgenticPatternType.HIERARCHICAL_P2P_A2A,
    "C4": AgenticPatternType.HIERARCHICAL_CROSS_MODEL,
    "CR": AgenticPatternType.CRITIC_REFLECTION_LOOP,
    # Prompt / persona variance on C1 (requirements 8 and 10): baseline is C1 = standard prompt,
    # Balanced-Engineer persona.
    "C1R": AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A,   # ReAct prompt (tool-using Thought/Action/Observation)
    "C1F": AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A,   # few-shot prompt
    "C1S": AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A,   # Strict-Architect persona
    "C1P": AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A,   # Permissive-Prototyper persona
}

VARIANT_OPTIONS = {
    "C1R": {"prompt_style": "react"},
    "C1F": {"prompt_style": "few_shot"},
    "C1S": {"persona": PersonaType.STRICT_ARCHITECT.value},
    "C1P": {"persona": PersonaType.PERMISSIVE_PROTOTYPER.value},
}


def _is_quota(text: str) -> bool:
    return "QuotaExhausted" in text or "RESOURCE_EXHAUSTED" in text or "429" in text


def run_cell(task_id: str, arch_id: str, persona: str, retries: int, judge) -> dict:
    prompt, relevant_symbols = TASKS[task_id]
    opts = VARIANT_OPTIONS.get(arch_id, {})
    persona = opts.get("persona", persona)
    prompt_style = opts.get("prompt_style", "standard")
    row = {"task_id": task_id, "arch": arch_id, "architecture": ARCHS[arch_id].value, "persona": persona,
           "prompt_style": prompt_style, "model": settings.DEFAULT_MODEL,
           "coder_model": settings.C4_CODER_MODEL if arch_id == "C4" else settings.DEFAULT_MODEL}
    start = time.time()
    try:
        orch = AutonomousEngineeringOrchestrator(
            pattern=ARCHS[arch_id], persona=persona,
            coder_prompt_style=prompt_style if ARCHS[arch_id] in (
                AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A, AgenticPatternType.HIERARCHICAL_CROSS_MODEL) else None,
        )
        m = None
        for attempt in range(1, retries + 1):
            try:
                m = orch.execute_requirement(prompt, target_repo="sample_repo")
                break
            except Exception as exc:
                if ("503" in str(exc) or "UNAVAILABLE" in str(exc)) and attempt < retries:
                    print(f"  provider overloaded (attempt {attempt}); retrying in {30 * attempt}s", flush=True)
                    time.sleep(30 * attempt)
                    continue
                raise
        row["attempts"] = attempt
        cov = (m.quality or {}).get("coverage") or {}
        chunks = (m.quality or {}).get("retrieved_chunks") or []
        row.update({
            "run_id": m.run_id,
            "success": m.success,
            "tests_passed": m.tests_passed,
            "tests_total": m.tests_total,
            "recovery_cycles": m.failure_recovery_cycles,
            "total_time_s": m.total_execution_time_seconds,
            "prompt_tokens": m.total_prompt_tokens,
            "completion_tokens": m.total_completion_tokens,
            "total_tokens": m.total_tokens,
            "cost_usd": m.total_cost_usd,
            "a2a_messages": m.a2a_message_count,
            "tool_calls": m.tool_call_count,
            "is_estimated": m.is_measurement_estimated,
            "error": m.error_details,
            "regressions": global_blackboard.state.metadata.get("regressions", []),
            "coverage_percent": cov.get("percent_covered") if cov.get("success") else None,
            "retrieval": retrieval_quality(chunks, [CALC], relevant_symbols) if chunks else None,
            "manager_decisions": [d.action.value for d in global_blackboard.state.manager_decisions],
            "diagnoses": [d.category.value for d in global_blackboard.state.failure_diagnoses],
            "agents": [
                {"agent": b.agent_id, "latency_s": b.latency_seconds, "tokens": b.total_tokens,
                 "llm_calls": b.llm_calls, "cost_usd": b.estimated_cost_usd}
                for b in m.agent_breakdowns
            ],
        })
        if m.error_details and _is_quota(m.error_details):
            row["excluded_from_stats"] = True
        elif judge is not None and m.run_id:
            try:
                v = judge.judge()
                row["judge"] = v.model_dump(mode="json")
            except Exception as exc:  # judge failure must not invalidate the run itself
                row["judge_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    except Exception as exc:
        row.update({"success": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "total_time_s": round(time.time() - start, 2)})
        row["excluded_from_stats"] = _is_quota(row["error"]) or "503" in row["error"]
        traceback.print_exc()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--archs", nargs="*", default=["C1", "C2", "C3", "C4"])
    parser.add_argument("--persona", default=PersonaType.BALANCED_ENGINEER.value)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Discard previous results")
    parser.add_argument("--rerun-failed", action="store_true", help="Re-run cells that were excluded (quota/503)")
    parser.add_argument("--out", default=str(ROOT / "sandbox_workspace" / "live_evaluation_results.json"))
    args = parser.parse_args()

    assert settings.EXECUTION_MODE == "live", "Live evaluation requires EXECUTION_MODE=live"
    out_path = Path(args.out)
    results = [] if args.fresh or not out_path.exists() else json.loads(out_path.read_text(encoding="utf-8"))
    done = {(r["task_id"], r["arch"]) for r in results
            if not (args.rerun_failed and r.get("excluded_from_stats"))}

    judge = None
    if not args.no_judge:
        from agents.google_adk.judge_agent import GoogleADKJudgeAgent
        judge = GoogleADKJudgeAgent()

    for task_id in args.tasks:
        for arch_id in args.archs:
            if (task_id, arch_id) in done:
                continue
            print(f"\n=== {task_id}/{arch_id}: {ARCHS[arch_id].value} ===", flush=True)
            row = run_cell(task_id, arch_id, args.persona, args.retries, judge)
            results = [r for r in results if (r["task_id"], r["arch"]) != (task_id, arch_id)] + [row]
            out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            print(json.dumps({k: row.get(k) for k in
                              ("run_id", "success", "tests_passed", "tests_total", "recovery_cycles",
                               "total_time_s", "total_tokens", "coverage_percent", "error")}, default=str), flush=True)
            if row.get("judge"):
                print(f"  judge overall: {row['judge'].get('overall')}", flush=True)
            if row.get("excluded_from_stats") and _is_quota(row.get("error") or ""):
                print("  QUOTA EXHAUSTED - stopping campaign; re-run later to resume.", flush=True)
                print(f"\nSaved {len(results)} rows to {out_path}")
                return

    print(f"\nSaved {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
