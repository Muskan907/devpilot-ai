# Multi-Agent AI System for Autonomous Software Engineering: Converting Developer Requirements into Validated Code Changes

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

All runs below made real calls to `gemini-3.5-flash`; measurements come from provider-reported token usage and wall-clock timing recorded by the usage ledger.

### 7.1 Per-run results

| Task | Configuration | Success | Tests | Repair cycles | Time (s) | Tokens | Coverage % | Judge | Retrieval P@5 | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | C1 Hierarchical + Blackboard + A2A | Yes | 10/10 | 0 | 39.025 | 7186 | 96.7 | 5.00 | 0.20 |  |
| T1 | C2 Sequential pipeline | Yes | 11/11 | 0 | 25.553 | 2658 | 100.0 | 5.00 | - |  |
| T1 | C3 Hierarchical, peer-to-peer messaging | Yes | 9/9 | 0 | 32.364 | 6978 | 96.7 | 5.00 | 0.20 |  |
| T1 | C4 Cross-model (local Ollama coder + embeddings) | Yes | 13/13 | 0 | 62.794 | 6240 | 97.1 | - | 0.40 | Judge call failed (provider overload); run itself valid |
| T2 | C1 Hierarchical + Blackboard + A2A | Yes | 8/8 | 0 | 31.032 | 5836 | 96.7 | 5.00 | 0.20 |  |
| T2 | C2 Sequential pipeline | Yes | 15/15 | 0 | 24.827 | 3111 | 66.0 | 4.50 | - |  |
| T2 | C3 Hierarchical, peer-to-peer messaging | Yes | 8/8 | 0 | 28.183 | 5606 | 96.7 | 5.00 | 0.20 |  |
| T2 | C4 Cross-model (local Ollama coder + embeddings) | Yes | 8/8 | 1 | 59.309 | 11113 | 96.5 | 4.80 | 0.40 |  |
| T3 | C1 Hierarchical + Blackboard + A2A | Yes | 8/8 | 0 | 29.462 | 5708 | 96.0 | 5.00 | 0.00 |  |
| T3 | C2 Sequential pipeline | Yes | 12/12 | 0 | 21.305 | 2899 | 68.4 | 4.50 | - |  |
| T3 | C3 Hierarchical, peer-to-peer messaging | Yes | 10/10 | 0 | 29.055 | 5747 | 96.0 | 5.00 | 0.00 |  |
| T3 | C4 Cross-model (local Ollama coder + embeddings) | Yes | 8/8 | 0 | 30.531 | 6107 | 96.0 | 5.00 | 0.40 |  |
| T4 | C1 Hierarchical + Blackboard + A2A | Yes | 13/13 | 0 | 40.809 | 6914 | 96.7 | 5.00 | 0.20 |  |
| T4 | C2 Sequential pipeline | Yes | 14/14 | 0 | 28.396 | 2662 | 100.0 | 4.50 | - |  |
| T4 | C3 Hierarchical, peer-to-peer messaging | Yes | 13/13 | 0 | 137.039 | 6802 | 96.5 | 5.00 | 0.20 |  |
| T4 | C4 Cross-model (local Ollama coder + embeddings) | Yes | 11/11 | 1 | 91.702 | 12437 | 97.1 | 5.00 | 0.40 |  |
| T5 | C1 Hierarchical + Blackboard + A2A | Yes | 8/8 | 0 | 40.498 | 6406 | 96.3 | - | 0.20 | Judge call failed (provider overload); run itself valid |
| T5 | C2 Sequential pipeline | Yes | 14/14 | 0 | 26.722 | 2413 | 100.0 | 4.50 | - |  |
| T5 | C3 Hierarchical, peer-to-peer messaging | Yes | 18/18 | 0 | 26.943 | 5790 | 96.3 | 5.00 | 0.20 |  |
| T5 | C4 Cross-model (local Ollama coder + embeddings) | Yes | 8/8 | 0 | 50.245 | 5351 | 96.5 | 4.75 | 0.40 |  |
| T6 | C1 Hierarchical + Blackboard + A2A | Yes | 9/9 | 0 | 23.873 | 5118 | 96.0 | 5.00 | 0.00 |  |
| T6 | C2 Sequential pipeline | Yes | 16/16 | 0 | 19.968 | 2388 | 100.0 | 5.00 | - |  |
| T6 | C3 Hierarchical, peer-to-peer messaging | Yes | 9/9 | 0 | 28.234 | 5570 | 96.0 | - | 0.00 | Judge call failed (provider overload); run itself valid |
| T6 | C4 Cross-model (local Ollama coder + embeddings) | Yes | 9/9 | 0 | 84.265 | 5917 | 96.3 | - | 0.40 | Judge call failed (provider overload); run itself valid |
| T7 | C1 Hierarchical + Blackboard + A2A | Yes | 11/11 | 1 | 57.521 | 11262 | 96.0 | 5.00 | 0.00 |  |
| T7 | C2 Sequential pipeline | No | 20/22 | 0 | 33.804 | 3373 | 75.5 | 3.75 | - | Sequential single pass failed: tests_all_passed=False, regressions=0, audit_passed=True |
| T7 | C3 Hierarchical, peer-to-peer messaging | Yes | 10/10 | 0 | 32.474 | 6020 | 96.3 | 5.00 | 0.00 |  |
| T7 | C4 Cross-model (local Ollama coder + embeddings) | Yes | 10/10 | 1 | 71.092 | 10522 | 96.3 | 5.00 | 0.40 |  |
| T8 | C1 Hierarchical + Blackboard + A2A | Yes | 8/8 | 0 | 37.088 | 5875 | 96.4 | - | 0.20 | Judge call failed (provider overload); run itself valid |
| T8 | C2 Sequential pipeline | No | 0/1 | 0 | 24.258 | 2820 | - | - | - | Sequential single pass failed: tests_all_passed=False, regressions=0, audit_passed=False |
| T8 | C3 Hierarchical, peer-to-peer messaging | Yes | 8/8 | 0 | 37.522 | 5536 | 96.3 | 5.00 | 0.20 |  |
| T8 | C4 Cross-model (local Ollama coder + embeddings) | Yes | 8/8 | 1 | 71.184 | 10682 | 96.8 | 5.00 | 0.40 |  |
| T9 | C1 Hierarchical + Blackboard + A2A | Yes | 10/10 | 0 | 48.193 | 6268 | 96.5 | 5.00 | 0.00 |  |
| T9 | C2 Sequential pipeline | Yes | 14/14 | 0 | 43.349 | 3496 | 60.0 | 5.00 | - |  |
| T9 | C3 Hierarchical, peer-to-peer messaging | Yes | 11/11 | 0 | 51.243 | 6543 | 71.2 | 5.00 | 0.00 |  |
| T9 | C4 Cross-model (local Ollama coder + embeddings) | Yes | 9/9 | 0 | 36.889 | 6179 | 81.1 | 4.50 | 0.40 |  |
| T10 | C1 Hierarchical + Blackboard + A2A | Yes | 11/11 | 0 | 75.401 | 5983 | 96.5 | 5.00 | 0.00 |  |
| T10 | C2 Sequential pipeline | Yes | 14/14 | 0 | 32.559 | 3196 | 77.8 | 4.25 | - |  |
| T10 | C3 Hierarchical, peer-to-peer messaging | Yes | 10/10 | 0 | 58.168 | 6321 | 96.0 | 5.00 | 0.00 |  |
| T10 | C4 Cross-model (local Ollama coder + embeddings) | Yes | 11/11 | 0 | 62.302 | 6102 | 80.7 | 5.00 | 0.20 |  |

### 7.2 Comparison by architecture

| Configuration | Runs | Success | Avg time (s) | Avg tokens | Avg cost ($) | Avg repair cycles | Coverage % | Judge /5 | Retrieval P@5 / MRR |
|---|---|---|---|---|---|---|---|---|---|
| C1 Hierarchical + Blackboard + A2A | 10 | 10/10 (100%) | 42.3 | 6656 | 0.00143 | 0.1 | 96.4 | 5.00 | 0.10 / 0.33 |
| C2 Sequential pipeline | 10 | 8/10 (80%) | 28.1 | 2902 | 0.00068 | 0.0 | 83.1 | 4.56 | - / - |
| C3 Hierarchical, peer-to-peer messaging | 10 | 10/10 (100%) | 46.1 | 6091 | 0.00131 | 0.0 | 93.8 | 5.00 | 0.10 / 0.33 |
| C4 Cross-model (local Ollama coder + embeddings) | 10 | 10/10 (100%) | 62.0 | 8065 | 0.00129 | 0.4 | 93.4 | 4.88 | 0.38 / 0.78 |

### 7.3 Per-agent response time (C1)

| Agent | Avg latency (s) | Avg tokens | Avg LLM calls |
|---|---|---|---|
| CrewAI:PlannerAgent | 14.28 | 3035 | 1.0 |
| GoogleADK:CoderAgent | 11.78 | 1566 | 1.0 |
| GoogleADK:TestGeneratorAgent | 10.88 | 1555 | 1.0 |
| GoogleADK:DebuggerAgent | 12.69 | 3487 | 1.0 |
| MCP:TestRunner | 0.91 | 0 | 0.0 |
| CrewAI:ManagerAgent | 5.65 | 1503 | 1.0 |
| GoogleADK:RepositoryRAGAgent | 0.09 | 0 | 0.0 |
| CrewAI:AuditorAgent | 0.00 | 0 | 0.0 |

### 7.4 Prompt and persona variance (C1, tasks T1, T2)

Same architecture (C1), same tasks; only the Coder prompt style or the team persona changes.

| Variant | Prompt style | Persona | Runs | Success | Avg time (s) | Avg tokens | Avg cost ($) | Repair cycles | Coverage % | Judge /5 |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | Standard | Balanced Engineer | 2 | 2/2 | 35.0 | 6511 | 0.00145 | 0.0 | 96.7 | 5.00 |
| C1R | ReAct (tool-using) | Balanced Engineer | 2 | 2/2 | 58.3 | 10186 | 0.00179 | 0.5 | 96.7 | 4.60 |
| C1F | Few-shot | Balanced Engineer | 2 | 2/2 | 43.4 | 6584 | 0.00140 | 0.0 | 96.7 | 5.00 |
| C1S | Standard | Strict Architect | 2 | 2/2 | 78.6 | 7730 | 0.00174 | 0.0 | 76.6 | 5.00 |
| C1P | Standard | Permissive Prototyper | 2 | 2/2 | 47.3 | 5994 | 0.00127 | 0.0 | 96.7 | 5.00 |

All variants succeeded. ReAct used the most tokens (extra tool-call turns to read files and tests) and needed the only repair cycle; few-shot cost about the same as the standard prompt (slightly more time) with no loss in judge score. The Strict Architect persona made the team write the most tests (13-14 vs 8-10) but took longest, and its T2 run measured lower line coverage (56%). The Permissive Prototyper used the fewest tokens and lowest cost of all variants. With two tasks per variant these are indicative, not statistically significant, differences.


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
