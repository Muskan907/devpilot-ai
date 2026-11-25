"""DevPilot AI - Streamlit dashboard (cloud demo).

Four tabs:
  Run           submit a requirement and watch the CrewAI + Google ADK agents work, then approve
  Architecture  the multi-agent design and the C1-C4 configurations that were compared
  Evaluation    the 48 live runs (10 tasks x C1-C4 + prompt/persona variants)
  Codebase      the target repository and RAG code search
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from core.a2a import global_a2a_bus  # noqa: E402
from core.blackboard import global_blackboard  # noqa: E402
from core.hitl import hitl_gatekeeper  # noqa: E402
from core.observability.tool_validation import tool_validation_stats  # noqa: E402
from mcp_servers.repo_tool import mcp_repo_client  # noqa: E402
from orchestration import AutonomousEngineeringOrchestrator, AgenticPatternType, PersonaType  # noqa: E402

st.set_page_config(page_title="DevPilot AI", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Access gate (APP_PASSWORD secret). Skipped when unset, e.g. local development.
# ---------------------------------------------------------------------------
_APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
if _APP_PASSWORD and not st.session_state.get("authed"):
    st.title("⚡ DevPilot AI")
    _pw = st.text_input("Access password", type="password")
    if _pw and _pw == _APP_PASSWORD:
        st.session_state["authed"] = True
        st.rerun()
    elif _pw:
        st.error("Incorrect password.")
    st.stop()

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1250px;}
      .dp-hero {border-radius: 14px; padding: 18px 22px; margin-bottom: 10px;
                background: linear-gradient(120deg, #0F2A44 0%, #134E5E 100%); color: #fff;}
      .dp-hero h1 {font-size: 1.7rem; margin: 0; color: #fff;}
      .dp-hero p {margin: 4px 0 10px 0; color: #CFE3EA; font-size: 0.95rem;}
      .dp-chip {display: inline-block; padding: 3px 10px; margin: 2px 6px 2px 0; border-radius: 999px;
                font-size: 0.78rem; background: rgba(255,255,255,0.14); color: #fff;}
      .dp-step {border: 1px solid rgba(128,128,128,0.25); border-radius: 10px; padding: 10px 12px; height: 100%;}
      .dp-step b {display: block; font-size: 0.92rem;}
      .dp-step span {font-size: 0.78rem; opacity: 0.75;}
      div[data-testid="stMetricValue"] {font-size: 1.45rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="dp-hero">
      <h1>⚡ DevPilot AI</h1>
      <p>A team of AI agents that turns a developer requirement into a tested, reviewed code change.</p>
      <span class="dp-chip">CrewAI: Planner · Manager · Auditor</span>
      <span class="dp-chip">Google ADK: RAG · Coder · Tester · Debugger · Judge</span>
      <span class="dp-chip">Blackboard + A2A</span>
      <span class="dp-chip">MCP tools</span>
      <span class="dp-chip">Model: {settings.DEFAULT_MODEL}</span>
      <span class="dp-chip">Mode: {settings.EXECUTION_MODE}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

ARCH_OPTIONS = {
    "C1 · Hierarchical + Blackboard (proposed)": AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A,
    "C2 · Sequential pipeline": AgenticPatternType.SEQUENTIAL_BASELINE,
    "C3 · Peer-to-peer messaging (no Blackboard)": AgenticPatternType.HIERARCHICAL_P2P_A2A,
    "Critic-reflection loop (extra pattern)": AgenticPatternType.CRITIC_REFLECTION_LOOP,
}
PERSONAS = {
    "Balanced Engineer": PersonaType.BALANCED_ENGINEER,
    "Strict Architect": PersonaType.STRICT_ARCHITECT,
    "Permissive Prototyper": PersonaType.PERMISSIVE_PROTOTYPER,
}
PROMPT_STYLES = {"Standard": "standard", "ReAct (tool-using)": "react", "Few-shot": "few_shot"}
PRESETS = {
    "Custom requirement": "",
    "T1 · standard deviation": "Add a standard_deviation(numbers, sample=False) method to AdvancedCalculator in src/calculator.py. It must raise ValueError for an empty list, and for sample=True with fewer than two values.",
    "T2 · median": "Add a median(numbers) method to AdvancedCalculator in src/calculator.py that returns the middle value (average of the two middle values for even-length input) and raises ValueError for an empty list.",
    "T3 · safe percentage": "Add a safe_percentage(part, whole) method to AdvancedCalculator in src/calculator.py that returns part / whole * 100 rounded to 2 decimals and raises ValueError when whole is 0.",
    "T5 · factorial": "Add a factorial(n) method to AdvancedCalculator in src/calculator.py that returns n! for non-negative integers and raises ValueError for negative numbers or non-integers.",
    "T6 · modulo": "Add a modulo(a, b) method to AdvancedCalculator in src/calculator.py that returns a % b and raises ZeroDivisionError with the message 'Cannot take modulo by zero' when b is 0.",
    "T10 · clamp": "Add a clamp(value, low, high) method to AdvancedCalculator in src/calculator.py that limits value to the range [low, high] and raises ValueError when low is greater than high.",
}
RESULTS_PATH = settings.SANDBOX_ROOT / "live_evaluation_results.json"
CONFIG_LABELS = {
    "C1": "C1 Hierarchical + Blackboard", "C2": "C2 Sequential", "C3": "C3 Peer-to-peer",
    "C4": "C4 Cross-model (Ollama coder)",
    "C1R": "C1 + ReAct prompt", "C1F": "C1 + few-shot prompt",
    "C1S": "C1 + Strict persona", "C1P": "C1 + Permissive persona",
}

for key, default in (("metrics", None), ("judge", None), ("rag_run_id", None), ("compare_rows", [])):
    st.session_state.setdefault(key, default)


def short_agent(agent_id: str) -> str:
    return agent_id.split(":")[-1].replace("Agent", "")


def run_pipeline(prompt: str, pattern, persona, prompt_style):
    global_blackboard.reset()
    tool_validation_stats.reset()
    orch = AutonomousEngineeringOrchestrator(
        pattern=pattern, persona=persona,
        coder_prompt_style=prompt_style if pattern == AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A else None,
    )
    return orch.execute_requirement(prompt, target_repo="sample_repo")


def friendly_error(err: str) -> str:
    if "QUOTA" in err.upper() or "RESOURCE_EXHAUSTED" in err or "429" in err:
        return "The Gemini API daily quota for this key is used up. Update GEMINI_API_KEY in the app secrets and try again."
    if "503" in err or "UNAVAILABLE" in err:
        return "Gemini is temporarily overloaded (HTTP 503). Wait a minute and run again."
    return err


tab_run, tab_arch, tab_eval, tab_code = st.tabs(["🚀 Run", "🏗️ Architecture", "📊 Evaluation", "📚 Codebase"])

# =============================================================================
# RUN
# =============================================================================
with tab_run:
    left, right = st.columns([3, 2], gap="large")
    with left:
        preset = st.selectbox("Example task", list(PRESETS), index=1, key="sel_preset")
        prompt = st.text_area("Requirement", value=PRESETS[preset], height=110,
                              placeholder="Describe the change you want in the sample calculator repository...")
    with right:
        arch_label = st.selectbox("Architecture", list(ARCH_OPTIONS), key="sel_arch")
        c_a, c_b = st.columns(2)
        persona_label = c_a.selectbox("Persona", list(PERSONAS), key="sel_persona")
        style_label = c_b.selectbox("Coder prompt", list(PROMPT_STYLES), key="sel_style",
                                    disabled=ARCH_OPTIONS[arch_label] != AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A)
        st.caption("C4 (local Ollama coder) needs a local model server, so it is not offered in the cloud demo; "
                   "its results are in the Evaluation tab.")

    steps = [("1 · RAG", "ADK retrieves code"), ("2 · Plan", "CrewAI Planner"), ("3 · Code", "ADK Coder"),
             ("4 · Test", "ADK Tester + MCP pytest"), ("5 · Repair", "Debugger → Manager re-plans"),
             ("6 · Audit", "CrewAI Auditor"), ("7 · Approve", "Human sign-off")]
    for col, (t, s) in zip(st.columns(len(steps)), steps):
        col.markdown(f'<div class="dp-step"><b>{t}</b><span>{s}</span></div>', unsafe_allow_html=True)
    st.write("")

    if st.button("Run the agents", type="primary", key="btn_run_pipeline", disabled=not prompt.strip()):
        st.session_state.judge = None
        with st.status("Agents working: retrieving code, planning, coding, testing...", expanded=False) as status:
            try:
                m = run_pipeline(prompt.strip(), ARCH_OPTIONS[arch_label], PERSONAS[persona_label],
                                 PROMPT_STYLES[style_label])
                st.session_state.metrics = m
                status.update(label=("Run finished: all checks passed" if m.success else "Run finished with problems"),
                              state="complete" if m.success else "error")
            except Exception as exc:  # noqa: BLE001
                st.session_state.metrics = None
                status.update(label="Run failed", state="error")
                st.error(friendly_error(f"{type(exc).__name__}: {exc}"))

    m = st.session_state.metrics
    if m is None:
        st.info("Pick an example task (or write your own), choose an architecture, and press **Run the agents**. "
                "A run takes about 30-60 seconds and makes 3-5 Gemini calls.")
    else:
        bb = global_blackboard.state
        if m.error_details and not m.success:
            st.error(friendly_error(m.error_details))
        k = st.columns(6)
        k[0].metric("Result", "✅ Passed" if m.success else "❌ Failed")
        k[1].metric("Tests", f"{m.tests_passed}/{m.tests_total}")
        k[2].metric("Time", f"{m.total_execution_time_seconds:.1f} s")
        k[3].metric("Tokens", f"{m.total_tokens:,}")
        k[4].metric("Cost", f"${m.total_cost_usd:.5f}")
        k[5].metric("Repair cycles", m.failure_recovery_cycles)
        cov = (m.quality or {}).get("coverage") or {}
        st.caption(f"Run `{m.run_id}` · {m.architecture_pattern} · {m.persona} · "
                   f"coverage {cov.get('percent_covered', '-')}% · {m.a2a_message_count} A2A messages · "
                   f"{m.tool_call_count} tool calls")

        t_diff, t_tests, t_agents, t_plan, t_msgs, t_approve = st.tabs(
            ["Code change", "Tests", "Agents & timing", "Plan & decisions", "A2A messages", "Approve"])

        with t_diff:
            delivery = bb.final_delivery
            diff = delivery.unified_diff if delivery and delivery.unified_diff else ""
            if not diff and m.run_id:
                diff = mcp_repo_client.get_diff(m.run_id).get("diff", "")
            st.code(diff or "No changes produced.", language="diff")

        with t_tests:
            if bb.test_reports:
                rep = bb.test_reports[-1]
                st.markdown(f"**{rep.passed} of {rep.total_tests} tests passed** in {rep.execution_time_seconds:.2f} s "
                            f"(existing suite + tests generated by the ADK TestGenerator).")
                st.dataframe(pd.DataFrame([{"Test": i.test_node_id, "Result": "✅ passed" if i.outcome == "passed" else f"❌ {i.outcome}",
                                            "Message": (i.failure_message or "")[:200]} for i in rep.items]),
                             hide_index=True, width="stretch")
            regs = bb.metadata.get("regressions") or []
            st.markdown(f"**Regressions vs. original code:** {len(regs) or 'none'}")
            snap = tool_validation_stats.snapshot()
            st.caption(f"Tool-output validation: {snap['checked']} MCP results type-checked, "
                       f"{len(snap['violations'])} violations.")

        with t_agents:
            rows = [{"Agent": short_agent(b.agent_id), "Framework": b.framework, "Latency (s)": round(b.latency_seconds, 2),
                     "LLM calls": b.llm_calls, "Tokens": b.total_tokens, "Cost ($)": round(b.estimated_cost_usd, 6)}
                    for b in m.agent_breakdowns]
            if rows:
                df = pd.DataFrame(rows)
                c1, c2 = st.columns([3, 2])
                c1.dataframe(df, hide_index=True, width="stretch")
                c2.bar_chart(df.set_index("Agent")["Latency (s)"], horizontal=True, height=260)

            if st.button("Score this run with the LLM judge (1 extra call)", key="btn_judge"):
                try:
                    from agents.google_adk.judge_agent import GoogleADKJudgeAgent
                    with st.spinner("Judge (gemini-3.5-flash-lite) is scoring the run..."):
                        st.session_state.judge = GoogleADKJudgeAgent().judge()
                except Exception as exc:  # noqa: BLE001
                    st.error(friendly_error(str(exc)))
            v = st.session_state.judge
            if v is not None:
                st.markdown(f"**LLM-as-Judge overall: {v.overall:.2f} / 5** ({v.judge_model})")
                crit = [("Plan quality", v.plan_quality), ("Code correctness", v.code_correctness),
                        ("Code quality", v.code_quality), ("Test adequacy", v.test_adequacy)]
                if v.diagnosis_quality:
                    crit.append(("Diagnosis quality", v.diagnosis_quality))
                st.dataframe(pd.DataFrame([{"Criterion": n, "Score": c.score, "Why": c.justification} for n, c in crit]),
                             hide_index=True, width="stretch")

        with t_plan:
            if bb.analysis:
                st.markdown(f"**Requirement understood as:** {bb.analysis.title}. {bb.analysis.problem_statement}")
                if bb.analysis.acceptance_criteria:
                    st.markdown("**Acceptance criteria:**")
                    for c in bb.analysis.acceptance_criteria:
                        st.markdown(f"- `{c.id}` {c.description}")
            if bb.plan:
                st.markdown("**Plan (CrewAI Planner):**")
                st.dataframe(pd.DataFrame([{"Step": t.title, "Framework": t.assigned_framework, "Role": t.assigned_role,
                                            "Status": t.status.value} for t in bb.plan.tasks]),
                             hide_index=True, width="stretch")
            if bb.failure_diagnoses or bb.manager_decisions:
                st.markdown("**Failure-driven re-planning:**")
                for d in bb.failure_diagnoses:
                    st.warning(f"Cycle {d.cycle} · Debugger diagnosis: **{d.category.value}**: {d.root_cause}")
                for d in bb.manager_decisions:
                    st.info(f"Cycle {d.cycle} · Manager decision: **{d.action.value}**: {d.rationale}")
            else:
                st.caption("No failures this run, so no re-planning was needed.")
            if bb.security_audit:
                st.markdown(f"**Security audit (CrewAI Auditor):** "
                            f"{'passed ✅' if getattr(bb.security_audit, 'passed', False) else 'issues found ⚠️'}")
            st.caption(f"Blackboard version: v{bb.version}")

        with t_msgs:
            hist = global_a2a_bus.get_history()
            if hist:
                st.dataframe(pd.DataFrame([{"Time": e.timestamp.strftime("%H:%M:%S"), "From": e.sender_id,
                                            "To": e.recipient_id, "Type": e.message_type.value,
                                            "Signature": e.signature[:12] + "…"} for e in hist]),
                             hide_index=True, width="stretch")
            else:
                st.caption("No A2A messages recorded.")

        with t_approve:
            status_txt = bb.human_approval_status
            st.markdown(f"Approval status: **{status_txt.upper()}**")
            st.caption("Nothing is committed until a human approves. Approval commits the change on an "
                       "`agentic/*` branch of the sandbox copy.")
            c1, c2 = st.columns(2)
            notes = c1.text_input("Reviewer notes", value="Tests pass and the diff is minimal.")
            if c1.button("✅ Approve", key="btn_hitl_approve", disabled=not m.success):
                res = hitl_gatekeeper.approve_and_commit(run_id=m.run_id, approver="Reviewer", notes=notes,
                                                         target_repo="sample_repo")
                (st.success if res.get("success") else st.error)(res.get("message") or res.get("error") or "Done")
            reason = c2.text_input("Rejection reason", value="Does not meet the requirement.")
            if c2.button("❌ Reject", key="btn_hitl_reject"):
                hitl_gatekeeper.reject(reason)
                st.warning("Change rejected; nothing was committed.")

# =============================================================================
# ARCHITECTURE
# =============================================================================
with tab_arch:
    st.subheader("How the agents work together (C1)")
    st.graphviz_chart(
        """
        digraph G {
          rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
          edge [fontname="Helvetica", fontsize=9, color="#5B7083"];
          U [label="Developer\\nrequirement", fillcolor="#E8EEF4"];
          G [label="Guardrails\\n(injection, toxicity)", fillcolor="#FDE2E1"];
          subgraph cluster_crew { label="CrewAI"; style="rounded"; color="#134E5E";
            P [label="Planner", fillcolor="#D7EEF2"]; M [label="Manager", fillcolor="#D7EEF2"]; A [label="Auditor", fillcolor="#D7EEF2"]; }
          subgraph cluster_adk { label="Google ADK"; style="rounded"; color="#0F2A44";
            R [label="RAG", fillcolor="#E3E7F7"]; C [label="Coder", fillcolor="#E3E7F7"]; T [label="Tester", fillcolor="#E3E7F7"];
            D [label="Debugger", fillcolor="#E3E7F7"]; J [label="Judge", fillcolor="#E3E7F7"]; }
          B [label="Blackboard\\n(shared, versioned state)", shape=cylinder, fillcolor="#FFF4D6"];
          subgraph cluster_mcp { label="MCP tool servers"; style="rounded,dashed"; color="#8A6D1D";
            RT [label="repo_tool\\nsandbox · read · patch · diff · RAG", fillcolor="#FFF9EB"];
            TR [label="test_runner\\nsyntax · pytest · coverage", fillcolor="#FFF9EB"]; }
          H [label="Human\\napproval", fillcolor="#E2F5E6"];
          U -> G -> R -> P -> C -> T;
          T -> H [label="all pass"];
          T -> D [label="fail"]; D -> M [label="diagnosis"]; M -> P [label="re-plan"]; M -> C [label="retry fix"];
          T -> A [style=dashed]; A -> H;
          {R C T D} -> RT [style=dotted, arrowhead=none]; T -> TR [style=dotted, arrowhead=none];
          {P C T D M} -> B [style=dashed, arrowhead=none, color="#C9A227"];
        }
        """,
        width="stretch",
    )
    st.caption("Agents exchange signed A2A envelopes (CrewAI ↔ ADK) and read/write the shared Blackboard. "
               "Tools are reached only through MCP, and every tool result is type-checked by a validation callback.")

    st.subheader("Configurations compared (10 tasks each, live Gemini runs)")
    st.dataframe(pd.DataFrame([
        {"Config": "C1 (proposed)", "Architecture": "Hierarchical, Manager-led", "Messaging": "Blackboard + A2A",
         "Repair loop": "Yes: Debugger diagnoses, Manager re-plans", "Models": "gemini-3.5-flash"},
        {"Config": "C2", "Architecture": "Sequential pipeline, same agents and tools", "Messaging": "Direct hand-off",
         "Repair loop": "No", "Models": "gemini-3.5-flash"},
        {"Config": "C3", "Architecture": "Same roles, no Manager hub", "Messaging": "Peer-to-peer A2A only (no Blackboard)",
         "Repair loop": "Debugger routes fixes itself", "Models": "gemini-3.5-flash"},
        {"Config": "C4", "Architecture": "C1", "Messaging": "Blackboard + A2A",
         "Repair loop": "Yes", "Models": "Coder: local qwen2.5:1.5b (Ollama); RAG: nomic-embed-text"},
    ]), hide_index=True, width="stretch")
    st.markdown(
        "**Why C1:** it is the only design that classifies a failure and chooses a different remedy "
        "(fetch more context, fix the code, or stop). In the live runs C1 succeeded 10/10; the sequential C2 "
        "was cheapest but failed 2/10 because it cannot repair; C3 matched C1 on these tasks but has no global view "
        "for re-planning decisions."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Security")
        st.markdown("- Prompt-injection and toxic-language guardrails on input\n"
                    "- AST scan of generated code (`eval`, `os.system`, `subprocess` ...)\n"
                    "- Path guard: tools cannot leave the per-run sandbox\n"
                    "- Retry limit and no-progress stop rule\n"
                    "- Human approval before any commit; pushes only to `agentic/*`")
    with c2:
        st.subheader("Observability")
        st.markdown("- Callbacks: agent start/finish/error, tool start/end, guardrail checks\n"
                    "- Tool-output schema validation on every MCP call\n"
                    "- Usage ledger: tokens, latency and cost per agent\n"
                    "- SHA-256 signed A2A envelopes and chained provenance log\n"
                    "- Pydantic schemas for every artefact (plan, patch, report, verdict)")

# =============================================================================
# EVALUATION
# =============================================================================
with tab_eval:
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
    rows = [r for r in rows if r.get("arch") and not r.get("excluded_from_stats")]
    if not rows:
        st.info("No live evaluation results found.")
    else:
        df = pd.DataFrame([{
            "Task": r["task_id"], "Config": r["arch"], "Success": bool(r.get("success")),
            "Tests": f"{r.get('tests_passed')}/{r.get('tests_total')}", "Repair cycles": r.get("recovery_cycles") or 0,
            "Time (s)": r.get("total_time_s"), "Tokens": r.get("total_tokens"), "Cost ($)": r.get("cost_usd"),
            "Coverage %": r.get("coverage_percent"), "Judge /5": (r.get("judge") or {}).get("overall"),
            "Retrieval P@5": (r.get("retrieval") or {}).get("precision_at_k"),
        } for r in rows])
        main = df[df["Config"].isin(["C1", "C2", "C3", "C4"])]
        k = st.columns(5)
        k[0].metric("Live runs", len(df))
        k[1].metric("Success (C1-C4)", f"{int(main['Success'].sum())}/{len(main)}")
        k[2].metric("Avg time / run", f"{df['Time (s)'].mean():.1f} s")
        k[3].metric("Avg cost / run", f"${df['Cost ($)'].mean():.5f}")
        k[4].metric("Avg judge score", f"{df['Judge /5'].mean():.2f} / 5")

        def summarize(frame, configs):
            g = frame[frame["Config"].isin(configs)].groupby("Config")
            out = pd.DataFrame({
                "Runs": g.size(), "Success %": (g["Success"].mean() * 100).round(0),
                "Avg time (s)": g["Time (s)"].mean().round(1), "Avg tokens": g["Tokens"].mean().round(0),
                "Avg cost ($)": g["Cost ($)"].mean().round(5), "Repair cycles": g["Repair cycles"].mean().round(2),
                "Coverage %": g["Coverage %"].mean().round(1), "Judge /5": g["Judge /5"].mean().round(2),
            }).reindex([c for c in configs if c in g.groups])
            out.index = [CONFIG_LABELS.get(c, c) for c in out.index]
            return out

        st.subheader("Architecture comparison (10 tasks × 4 configurations)")
        arch = summarize(main, ["C1", "C2", "C3", "C4"])
        st.dataframe(arch, width="stretch")
        g1, g2, g3 = st.columns(3)
        g1.caption("Success rate (%)")
        g1.bar_chart(arch["Success %"], height=220)
        g2.caption("Average time per run (s)")
        g2.bar_chart(arch["Avg time (s)"], height=220)
        g3.caption("Average tokens per run")
        g3.bar_chart(arch["Avg tokens"], height=220)
        st.caption("C2 failed T7 and T8: its generated tests failed and it has no repair loop. "
                   "C4's local coder needed repair cycles more often but always recovered; its learned embeddings "
                   "gave the best retrieval (P@5 0.38 vs 0.10).")

        st.subheader("Prompt and persona variance (C1, tasks T1-T2)")
        var = summarize(df[df["Task"].isin(["T1", "T2"])], ["C1", "C1R", "C1F", "C1S", "C1P"])
        st.dataframe(var, width="stretch")

        st.subheader("Per-agent response time (C1, average over 10 runs)")
        lat = {}
        for r in rows:
            if r["arch"] == "C1":
                for a in r.get("agents") or []:
                    lat.setdefault(short_agent(a["agent"]), []).append(a.get("latency_s") or 0)
        if lat:
            lat_df = pd.DataFrame({"Avg latency (s)": {k_: sum(v) / len(v) for k_, v in lat.items()}}).sort_values(
                "Avg latency (s)", ascending=False)
            st.bar_chart(lat_df, horizontal=True, height=260)

        with st.expander("All runs"):
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button("Download results (JSON)", RESULTS_PATH.read_bytes(), "live_evaluation_results.json")

    st.subheader("Quick live comparison")
    st.caption("Runs one requirement through C1, C2 and C3 now (about 10-15 Gemini calls).")
    cmp_prompt = st.text_input("Requirement", value=PRESETS["T3 · safe percentage"], key="cmp_prompt")
    if st.button("Run C1, C2 and C3", key="btn_run_compare"):
        st.session_state.compare_rows = []
        for label in list(ARCH_OPTIONS)[:3]:
            with st.spinner(f"Running {label}..."):
                try:
                    mm = run_pipeline(cmp_prompt, ARCH_OPTIONS[label], PersonaType.BALANCED_ENGINEER, "standard")
                    st.session_state.compare_rows.append({
                        "Configuration": label, "Success": "✅" if mm.success else "❌",
                        "Tests": f"{mm.tests_passed}/{mm.tests_total}", "Time (s)": round(mm.total_execution_time_seconds, 1),
                        "Tokens": mm.total_tokens, "Cost ($)": round(mm.total_cost_usd, 5),
                        "Repair cycles": mm.failure_recovery_cycles})
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{label}: {friendly_error(str(exc))}")
                    break
    if st.session_state.compare_rows:
        st.dataframe(pd.DataFrame(st.session_state.compare_rows), hide_index=True, width="stretch")

# =============================================================================
# CODEBASE
# =============================================================================
with tab_code:
    st.caption("Target repository: `sample_repo` (a small Python calculator package with a pytest suite). "
               "Each run works on an isolated sandbox copy.")
    reindex = st.button("Re-index repository", key="btn_connect_repo")
    if st.session_state.rag_run_id is None or reindex:
        with st.spinner("Indexing the repository (AST chunks → vector store)..."):
            sb = mcp_repo_client.create_sandbox("sample_repo")
            if sb.get("success"):
                st.session_state.rag_run_id = sb["run_id"]
                st.session_state.rag_stats = mcp_repo_client.index_rag(sb["run_id"])
    stats = st.session_state.get("rag_stats") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Index status", "Ready ✅" if stats.get("success") else "Not indexed")
    c2.metric("Code chunks (classes / methods)", stats.get("total_chunks_scanned", 0))
    c3.metric("Embedding", "feature hashing (C1-C3)")

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Search the code (RAG)**")
        q = st.text_input("Question", value="where is the average of a list computed?", key="rag_q")
        top_k = st.slider("Results", 1, 8, 4)
        if st.button("Search", key="btn_rag_search"):
            res = mcp_repo_client.query_rag(query=q, run_id=st.session_state.rag_run_id, top_k=top_k)
            chunks = res.get("results", [])
            if not chunks:
                st.warning("No matching code found.")
            for i, ch in enumerate(chunks):
                with st.expander(f"{i + 1}. {ch.get('symbol_name')} · {ch.get('file_path')} "
                                 f"(lines {ch.get('start_line')}-{ch.get('end_line')})", expanded=i == 0):
                    st.code(ch.get("content", ""), language="python")
    with right:
        st.markdown("**Browse files**")
        files = []
        if st.session_state.rag_run_id:
            files = [f for f in (mcp_repo_client.list_files(st.session_state.rag_run_id).get("files") or [])
                     if str(f).endswith((".py", ".toml", ".md", ".txt"))]
        if files:
            sel = st.selectbox("File", files, index=files.index("src/calculator.py") if "src/calculator.py" in files else 0,
                               key="sel_file")
            content = mcp_repo_client.read_file(st.session_state.rag_run_id, sel).get("content", "")
            st.code(content, language="python" if sel.endswith(".py") else None)
