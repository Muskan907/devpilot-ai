"""Generates FINAL_REPORT_DRAFT.md with the live evaluation results filled in.

Usage: .venv\\Scripts\\python.exe scripts\\build_report.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "sandbox_workspace" / "live_evaluation_results.json"
OUT = ROOT / "FINAL_REPORT_DRAFT.md"

ARCH = {
    "C1": "C1 Hierarchical + Blackboard + A2A",
    "C2": "C2 Sequential pipeline",
    "C3": "C3 Hierarchical, peer-to-peer messaging",
    "C4": "C4 Cross-model (local Ollama coder + embeddings)",
    "CR": "Critic-reflection loop (extra pattern)",
}
MAIN = ("C1", "C2", "C3", "C4", "CR")
VARIANTS = {
    "C1": ("Standard", "Balanced Engineer"),
    "C1R": ("ReAct (tool-using)", "Balanced Engineer"),
    "C1F": ("Few-shot", "Balanced Engineer"),
    "C1S": ("Standard", "Strict Architect"),
    "C1P": ("Standard", "Permissive Prototyper"),
}


def _f(v, d=1):
    return "-" if v is None else f"{float(v):.{d}f}"


def results_section() -> str:
    rows = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else []
    all_rows = [r for r in rows if r.get("arch")]
    rows = sorted([r for r in all_rows if r["arch"] in MAIN], key=lambda r: (int(r["task_id"][1:]), r["arch"]))
    if not rows:
        return "_Live results pending: run `scripts/run_live_evaluation.py`._\n"

    model = rows[0].get("model", "")
    out = [f"All runs below made real calls to `{model}`; measurements come from provider-reported "
           f"token usage and wall-clock timing recorded by the usage ledger.\n"]

    out.append("### 7.1 Per-run results\n")
    out.append("| Task | Configuration | Success | Tests | Repair cycles | Time (s) | Tokens | Coverage % | Judge | Retrieval P@5 | Note |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        tests = f"{r.get('tests_passed')}/{r.get('tests_total')}" if r.get("run_id") else "-"
        note = " ".join((r.get("error") or "")[:90].replace("|", "/").split())
        if r.get("excluded_from_stats"):
            note = "Not completed: Gemini API quota exhausted (excluded from averages)"
        elif r.get("earlier_attempts"):
            note = (note + " Re-run after fixing a crash on invalid generated code.").strip()
        if r.get("judge_error") and not note:
            note = "Judge call failed (provider overload); run itself valid"
        out.append(
            f"| {r['task_id']} | {ARCH[r['arch']]} | {'Yes' if r.get('success') else 'No'} | {tests} | "
            f"{r.get('recovery_cycles', '-') if r.get('run_id') else '-'} | {r.get('total_time_s', '-')} | "
            f"{r.get('total_tokens', '-') if r.get('run_id') else '-'} | {_f(r.get('coverage_percent'))} | "
            f"{_f((r.get('judge') or {}).get('overall'), 2)} | {_f((r.get('retrieval') or {}).get('precision_at_k'), 2)} | {note} |"
        )

    out.append("\n### 7.2 Comparison by architecture\n")
    out.append("| Configuration | Runs | Success | Avg time (s) | Avg tokens | Avg cost ($) | Avg repair cycles | Coverage % | Judge /5 | Retrieval P@5 / MRR |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in rows:
        if not r.get("excluded_from_stats"):
            by[r["arch"]].append(r)
    for a in ("C1", "C2", "C3", "C4", "CR"):
        rs = by.get(a, [])
        if not rs:
            continue
        done = [r for r in rs if r.get("run_id")]
        ok = sum(1 for r in rs if r.get("success"))
        avg = lambda k: (sum(float(r.get(k) or 0) for r in done) / len(done)) if done else 0.0
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None
        cov = mean([r["coverage_percent"] for r in done if r.get("coverage_percent") is not None])
        jdg = mean([r["judge"]["overall"] for r in done if (r.get("judge") or {}).get("overall")])
        prec = mean([r["retrieval"]["precision_at_k"] for r in done if r.get("retrieval")])
        mrr = mean([r["retrieval"]["mrr"] for r in done if r.get("retrieval")])
        out.append(f"| {ARCH[a]} | {len(rs)} | {ok}/{len(rs)} ({100 * ok // len(rs)}%) | {avg('total_time_s'):.1f} | "
                   f"{avg('total_tokens'):.0f} | {avg('cost_usd'):.5f} | {avg('recovery_cycles'):.1f} | {_f(cov)} | "
                   f"{_f(jdg, 2)} | {_f(prec, 2)} / {_f(mrr, 2)} |")

    lat = defaultdict(list)
    for r in by.get("C1", []):
        for ag in r.get("agents") or []:
            lat[ag["agent"]].append((float(ag.get("latency_s") or 0), int(ag.get("tokens") or 0), int(ag.get("llm_calls") or 0)))
    if lat:
        out.append("\n### 7.3 Per-agent response time (C1)\n")
        out.append("| Agent | Avg latency (s) | Avg tokens | Avg LLM calls |")
        out.append("|---|---|---|---|")
        for ag, vals in sorted(lat.items(), key=lambda kv: -sum(v[0] for v in kv[1])):
            n = len(vals)
            out.append(f"| {ag} | {sum(v[0] for v in vals) / n:.2f} | {sum(v[1] for v in vals) / n:.0f} | "
                       f"{sum(v[2] for v in vals) / n:.1f} |")
    out.append(variance_section(all_rows))
    return "\n".join(out) + "\n"


def variance_section(all_rows) -> str:
    var_tasks = sorted({r["task_id"] for r in all_rows if r["arch"] in VARIANTS and r["arch"] != "C1"},
                       key=lambda t: int(t[1:]))
    if not var_tasks:
        return ""
    out = ["\n### 7.4 Prompt and persona variance (C1, tasks " + ", ".join(var_tasks) + ")\n",
           "Same architecture (C1), same tasks; only the Coder prompt style or the team persona changes.\n",
           "| Variant | Prompt style | Persona | Runs | Success | Avg time (s) | Avg tokens | Avg cost ($) | Repair cycles | Coverage % | Judge /5 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, (style, persona) in VARIANTS.items():
        rs = [r for r in all_rows if r["arch"] == a and r["task_id"] in var_tasks and not r.get("excluded_from_stats")]
        if not rs:
            continue
        n = len(rs)
        ok = sum(1 for r in rs if r.get("success"))
        avg = lambda k: sum(float(r.get(k) or 0) for r in rs) / n
        cov = [r["coverage_percent"] for r in rs if r.get("coverage_percent") is not None]
        jd = [r["judge"]["overall"] for r in rs if (r.get("judge") or {}).get("overall")]
        out.append(f"| {a} | {style} | {persona} | {n} | {ok}/{n} | {avg('total_time_s'):.1f} | {avg('total_tokens'):.0f} | "
                   f"{avg('cost_usd'):.5f} | {avg('recovery_cycles'):.1f} | {_f(sum(cov) / len(cov) if cov else None)} | "
                   f"{_f(sum(jd) / len(jd) if jd else None, 2)} |")
    out.append("\nAll variants succeeded. ReAct used the most tokens (extra tool-call turns to read files and tests) and "
               "needed the only repair cycle; few-shot cost about the same as the standard prompt (slightly more time) with no loss in judge score. "
               "The Strict Architect persona made the team write the most tests (13-14 vs 8-10) but took longest, and its T2 run "
               "measured lower line coverage (56%). The Permissive Prototyper used the fewest tokens and lowest cost "
               "of all variants. With two tasks per variant these are indicative, "
               "not statistically significant, differences.")
    return "\n".join(out)


TEMPLATE = """# Multi-Agent AI System for Autonomous Software Engineering: Converting Developer Requirements into Validated Code Changes

**Final Project Report**
Muskan Patni (230534), Vaishnavi Singh (230508)
Project Guide: Prof. Pranshu Tiwari, BML Munjal University

---

## Abstract

Most software work changes an existing codebase, where the difficulty is modifying one behaviour without breaking another. We present DevPilot AI, a multi-agent system that converts a natural-language developer requirement into a validated code change on an isolated branch, together with the evidence behind it (tests, regression check, diff, security audit), and hands it to a human for approval. The system combines two agentic frameworks: CrewAI (Manager/Planner, Auditor) and Google ADK (Repository/RAG, Coder, Test, Debugger). Agents share state through a versioned Blackboard, exchange typed and signed A2A envelopes across frameworks, and use tools through two Model Context Protocol (MCP) servers. When tests fail, the Debugger classifies the failure and the Manager decides whether to retrieve more context, request a code or test fix, or stop, under a retry limit. We compare the hierarchical Blackboard design (C1) with a sequential pipeline (C2), peer-to-peer messaging without a Blackboard (C3) and a cross-model variant with a local Ollama coder and learned embeddings (C4) using live runs, reporting success, end-to-end and per-agent latency, tokens, cost, test coverage, retrieval quality and LLM-as-Judge scores.

## 1. Problem Statement and Objectives

A request such as "Add password-reset functionality and make sure existing functionality is not affected" requires reading the repository, locating relevant files and tests, editing code, running tests and recovering from failures. A single model call produces plausible code but no evidence that it works or that nothing regressed.

**Objectives**
1. Convert natural-language requirements into validated code changes with minimal human intervention.
2. Ground every change in the actual repository using retrieval-augmented generation.
3. Validate changes with tests and detect regressions against a baseline.
4. Recover from failures through diagnosis and failure-driven re-planning.
5. Keep a mandatory human approval checkpoint before any commit.
6. Study how architecture choice affects reliability, latency and cost.

## 2. Literature Review

| Work / product | Approach | Limitation (gap) |
|---|---|---|
| SWE-bench [1] | Benchmark of real GitHub issues | G1: evaluates models; does not provide a validated change workflow |
| SWE-agent [2] | Single agent with an agent-computer interface | G2: one agent; no explicit diagnosis/re-planning roles |
| MetaGPT [3] | Multi-agent roles via standard operating procedures | G3: mostly greenfield generation, weak grounding in existing repositories |
| ReAct [5] | Interleaved reasoning and acting | G2: single-agent loop, no shared state |
| RAG [7] | Retrieval-augmented generation | G3: text retrieval, not code-structure aware |
| Blackboard model [9] | Shared-state problem solving | G4: pre-LLM, not evaluated for LLM agent teams |
| AI coding assistants (e.g. GitHub Copilot) | IDE suggestions and chat agents | G5: limited per-step evidence and cost/latency transparency |

**Gap.** No open system combines repository-grounded multi-agent roles, test-verified changes with regression checks, failure-driven re-planning, a human approval gate, and measured per-agent cost and latency.

## 3. Agent Requirements

| Req. | Requirement | Source | Agent / component |
|---|---|---|---|
| R1 | Turn a plain-language request into tasks and acceptance criteria | Problem statement | Planner (CrewAI) |
| R2 | Ground changes in the actual repository | G3 | Repository/RAG agent (ADK, AST chunking) |
| R3 | Edit code only in a sandbox / agentic branch | Problem statement | Coder (ADK) + MCP repository tool |
| R4 | Prove correctness and detect regressions | G1, G5 | Test agent (ADK) + MCP test runner |
| R5 | Diagnose failures; decide more context / fix / stop | G2 | Debugger (ADK) + Manager (CrewAI) |
| R6 | Shared, consistent view of the run | G4 | Blackboard + A2A envelopes |
| R7 | Human approval before any commit | Problem statement | HITL gate |
| R8 | Measure per-agent latency, tokens and cost | G5 | Callbacks + usage ledger |

## 4. System Architecture

**Chosen architecture: hierarchical.** The CrewAI Manager sits at the top; specialists sit below it; all agents read and write a shared Blackboard.

Flow of one run:
1. Input guardrails (prompt injection, toxic language).
2. MCP repository tool creates an isolated sandbox (or clones the repository) and an `agentic/*` branch.
3. Baseline test run on the untouched code (used for regression detection).
4. Repository/RAG agent indexes the code into AST-level chunks (functions, classes) in ChromaDB and retrieves the relevant ones.
5. Planner (CrewAI) produces a structured `RequirementAnalysis` (acceptance criteria) and `ExecutionPlan`.
6. Manager delegates coding to the ADK Coder via an A2A envelope; the Coder edits the target file in the sandbox.
7. Manager delegates testing to the ADK Test agent, which generates targeted pytest tests from the acceptance criteria and runs them with the existing suite, reporting regressions.
8. Failure-driven re-planning loop (Section 4.1).
9. AST security audit of every changed file; Auditor prepares the delivery package (summary, diff, test summary, provenance hash).
10. Human approval gate; on approval the change is committed on the agentic branch only.

### 4.1 Failure-driven re-planning loop (PLAN, ACT, OBSERVE, REASON, RE-PLAN, ACT)

When tests fail, the Debugger (ADK) returns a structured `FailureDiagnosis`: a category (`missing_import`, `wrong_context`, `incorrect_logic`, `broken_test`, `syntax_error`, `environment`), root cause, target file, recommended action and optionally a corrected file. The Manager (CrewAI) then makes a `ManagerDecision`: `retrieve_more_context`, `request_code_fix`, `request_test_fix` or `stop`. Hard policy rules the LLM cannot override enforce a retry limit (3 cycles) and stop a run whose failure repeats unchanged. The Planner appends debug/verify tasks to the plan, the chosen action is executed, and tests re-run.

### 4.2 Agentic design patterns compared

| Config | What is varied | Question it answers |
|---|---|---|
| C1 | Hierarchical, Blackboard + A2A (primary) | Baseline performance of the proposed design |
| C2 | Sequential pipeline, same agents and tools, no repair loop | Does a manager-led design help over a straight pipeline? |
| C3 | Same roles, peer-to-peer A2A messaging, no Blackboard and no Manager hub | Which messaging approach gives better reliability and latency? |
| C4 | C1 with a local Ollama Coder (qwen2.5:1.5b) and learned Ollama embeddings (nomic-embed-text) for RAG | Which LLM suits which agent role? |

A generator-critic reflection loop is implemented as an additional agentic pattern.

In C3 every hand-off is a signed A2A envelope carrying the complete state the recipient needs (requirement, plan, patches, test report, diagnosis); before an agent acts, its working view is rebuilt only from the message it received, and the Debugger routes failures directly to the next peer.

## 5. Communication: Blackboard, A2A and MCP

**Blackboard.** A thread-safe, versioned shared state (`RLock`, monotonic version `v0..vN`, observer callbacks) holding the requirement, analysis, plan, patches, test reports, failure diagnoses, manager decisions, security audit, provenance and approval status.

**A2A.** Agents exchange `A2AEnvelope` messages (sender/recipient framework, correlation id, SHA-256 signature) of types TASK_DELEGATION, TASK_RESULT and REPLAN_REQUEST. Seven agents are registered: three CrewAI (Manager, Planner, Auditor) and four ADK (Coder, Test, Debugger, RAG). The bus is in-process; agents are not separate network services.

**MCP.** Two FastMCP servers: `mcp-repository-tool` (sandbox, clone, read/list, apply patch, Git branch/status/diff, RAG index/query, gated commit, PR publishing) and `mcp-test-runner` (syntax check, sandboxed pytest with timeout, structured per-test report).

**Structured output.** All inter-agent artefacts are Pydantic v2 models (requirements, plans, patches, test reports, diagnoses, decisions, metrics, provenance).

## 6. Security

* **Guardrails:** prompt-injection patterns; toxic/abusive language detection; AST scan of generated code for dangerous calls (`os.system`, `eval`, `subprocess`, ...); path guard preventing traversal outside the sandbox; pushes restricted to `agentic/*` branches.
* **Safeguards:** per-run isolated sandbox; human approval gate; retry limit and no-progress stop; quota errors fail the run instead of substituting output; secrets only through environment variables / cloud secrets.
* **Tool-output validation callbacks:** every MCP tool call (sandbox, read/list files, patch, syntax check, pytest, coverage) passes through a `validated_tool` wrapper that fires `on_tool_start/end` and checks the returned data against a Pydantic schema; malformed or out-of-range results are logged as `ToolOutputValidation` guardrail violations.
* **Callbacks and audit:** `on_agent_start/finish/error`, `on_tool_start/end`, `on_guardrail_check`; a per-agent usage ledger (tokens, latency, estimated flag); a SHA-256 chained provenance log.

## 7. Experimental Design and Results

**Design.** The main matrix is 10 change tasks (T1-T10: standard deviation, median, safe percentage, variance, factorial, modulo, compound interest, mode, n-th root, clamp) x 4 configurations = 40 runs, with the Balanced-Engineer persona; all 40 were completed live. Each task declares its relevant code (ground truth) for retrieval metrics. Variance comes from task content, from architecture (C1-C4), from the model (C4), and, in an additional 8 runs on C1, from the Coder prompt style (standard vs ReAct vs few-shot) and the team persona (Balanced vs Strict Architect vs Permissive Prototyper).

**Measures per run.** Success (all tests pass, no regressions, security audit passes); tests passed; repair cycles; end-to-end and per-agent latency; tokens and cost (per call, local models at $0); line coverage via the MCP coverage tool; retrieval precision@5 / recall / MRR against ground truth; and an LLM-as-Judge rubric score.

**LLM-as-Judge.** An ADK judge agent using a different model (gemini-3.5-flash-lite) than the generator scores plan quality, code correctness, code quality, test adequacy and (when present) diagnosis quality from 1 to 5 with justifications. Tests decide pass/fail; the judge rates what tests cannot measure.

**Offline verification.** 129 automated tests (simulation mode) cover failure injection: missing import, logic bug, regressions, syntax errors in generated code, retry-limit stop, quota exhaustion, guardrails, P2P messaging, coverage and retrieval metrics.

{RESULTS}

### 7.5 Validation approach

Correctness is decided by executable reference tests (existing and generated suites, plus a regression check against the untouched baseline), as the synopsis prefers. The LLM-as-Judge provides the supporting rubric score for plan, code and test quality.

## 8. Limitations

1. The Gemini free tier allows only 20 generate requests per model per project per day, so the 48 live runs were spread over several API projects; runs interrupted by quota were re-run and never counted. Five LLM-as-Judge calls failed with provider overload (HTTP 503) and those runs have no judge score.
2. With 10 runs per configuration (2 per prompt/persona variant) and mostly first-attempt successes, differences in success rate are not statistically significant; latency, cost, coverage and retrieval differences are more informative.
3. C4's local model (qwen2.5:1.5b) was limited by available laptop memory; a 7B coder did not fit.
4. A2A runs over an in-process bus rather than separate network services.
5. Code chunking and the AST security scan are Python-specific; the benchmark uses a single Python repository.
6. C1-C3 use deterministic hashing embeddings for RAG; learned embeddings are evaluated only in C4.

## 9. Team Contributions

| Member | Responsibilities |
|---|---|
| Muskan Patni (230534) | Repository/RAG agent; Coding agent; Testing/Validation agent; MCP tool layer; sandbox and Git integration; experiment runner; deployment |
| Vaishnavi Singh (230508) | Manager/Planner agent; orchestration and re-planning logic; Blackboard and A2A communication; Debugging/Review agent; observability, logging and tracing; HITL gate; system-level evaluation |
| Both | Literature review; task selection; integration; pilot testing; evaluation; documentation and presentation |

## 10. Conclusion

DevPilot AI shows that a hierarchical team of CrewAI and Google ADK agents, coordinated through a Blackboard and A2A messages and acting through MCP tools, can turn a plain-language requirement into a test-verified, regression-checked code change that a human approves. Its distinguishing feature is failure-driven re-planning: failures are classified and the Manager chooses a remedy within a retry limit. Every run is measured per agent from real provider data, and runs that cannot reach the model are reported as failures. Future work: more runs per configuration for statistical significance, larger local models for C4, multi-language repositories, and A2A agents deployed as separate network services.

## References

[1] C. E. Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", ICLR, 2024. arXiv:2310.06770.
[2] J. Yang et al., "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering," NeurIPS 37, 2024. arXiv:2405.15793.
[3] S. Hong et al., "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework," ICLR, 2024. arXiv:2308.00352.
[4] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed., Pearson, 2020.
[5] S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR, 2023. arXiv:2210.03629.
[6] J. Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," NeurIPS 35, 2022. arXiv:2201.11903.
[7] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 33, 2020. arXiv:2005.11401.
[8] Model Context Protocol, documentation and specification.
[9] H. P. Nii, "The Blackboard Model of Problem Solving and the Evolution of Blackboard Architectures," AI Magazine, vol. 7, no. 2, pp. 38-53, 1986.
[10] L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS 36, 2023. arXiv:2306.05685.
[11] CrewAI documentation; Google Agent Development Kit (ADK) documentation; Agent2Agent (A2A) protocol documentation.
"""


def main() -> None:
    OUT.write_text(TEMPLATE.replace("{RESULTS}", results_section()), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
