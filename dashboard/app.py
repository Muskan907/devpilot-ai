"""DevPilot AI - web front end (Streamlit, deployed on Streamlit Community Cloud).

Sections (top navigation):
  Home          landing page: what the system is, how it works, the agent team, headline results
  Studio        submit a requirement, watch the CrewAI + Google ADK agents work, review and approve
  Architecture  multi-agent design, C1-C4 configurations, security and observability
  Results       the 48 live evaluation runs
  Codebase      target repository browser and RAG code search
"""

import html as _html
import json
import os
import sys
from pathlib import Path

import altair as alt
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
from core.schemas.repository import RepositoryConfig  # noqa: E402
from orchestration import AutonomousEngineeringOrchestrator, AgenticPatternType, PersonaType  # noqa: E402

st.set_page_config(page_title="DevPilot AI · Autonomous Software Engineering", page_icon="⚡",
                   layout="wide", initial_sidebar_state="collapsed")


def html(markup: str) -> None:
    """Render raw HTML (lines are stripped so Markdown never treats them as code blocks)."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def esc(text) -> str:
    return _html.escape(str(text))


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --bg:#0B1020; --panel:#121935; --panel2:#18214A; --line:rgba(148,163,255,0.14); --text:#E7EAF6;
        --muted:#9AA3C7; --brand:#7C83FF; --brand2:#22D3EE; --ok:#34D399; --bad:#F87171; --warn:#FBBF24; }
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select { font-family: 'Inter', sans-serif !important; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], .stDeployButton,
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; visibility:hidden !important; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, rgba(124,131,255,0.18), transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,0.10), transparent 60%), var(--bg); }
.block-container { padding-top: 0.6rem !important; padding-bottom: 2rem; max-width: 1240px; }

/* navbar */
.dp-nav { display:flex; align-items:center; justify-content:space-between; padding:14px 4px 10px 4px; }
.dp-logo { display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.2rem; color:var(--text); }
.dp-logo-mark { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-size:1.05rem;
                background: linear-gradient(135deg, var(--brand), var(--brand2)); box-shadow:0 6px 22px rgba(124,131,255,0.45); }
.dp-nav-right { display:flex; gap:8px; align-items:center; }
.dp-pill { font-size:0.74rem; padding:5px 11px; border-radius:999px; border:1px solid var(--line); color:var(--muted); background:rgba(255,255,255,0.03); }
.dp-live { color:var(--ok); border-color:rgba(52,211,153,0.35); }
.dp-live::before { content:"●"; margin-right:6px; }

/* top navigation = tabs */
div[role="tablist"] { gap:4px !important; background:rgba(18,25,53,0.8); border:1px solid var(--line); border-radius:14px; padding:5px !important; width:fit-content; }
[data-testid="stTab"] { padding:8px 18px !important; border-radius:10px !important; cursor:pointer; transition:background .15s ease; }
[data-testid="stTab"] p { font-weight:600 !important; font-size:0.92rem !important; color:var(--muted) !important; margin:0 !important; }
[data-testid="stTab"]:hover { background:rgba(255,255,255,0.06); }
[data-testid="stTab"]:hover p { color:var(--text) !important; }
[data-testid="stTab"][aria-selected="true"] { background:linear-gradient(135deg, rgba(124,131,255,0.95), rgba(34,211,238,0.8)) !important; box-shadow:0 4px 16px rgba(124,131,255,0.35); }
[data-testid="stTab"][aria-selected="true"] p { color:#fff !important; }
.react-aria-SelectionIndicator, div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display:none !important; }
[data-testid="stTabPanel"], div[role="tabpanel"] { padding-top: 1.3rem; }

/* hero */
.dp-hero { padding:46px 8px 24px 8px; text-align:center; }
.dp-eyebrow { display:inline-block; font-size:0.78rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
              color:var(--brand2); background:rgba(34,211,238,0.08); border:1px solid rgba(34,211,238,0.25); padding:6px 14px; border-radius:999px; }
.dp-h1 { font-size:3.1rem; line-height:1.08; font-weight:800; margin:18px auto 14px auto; max-width:900px; color:var(--text); letter-spacing:-0.02em; }
.dp-grad { background: linear-gradient(90deg, var(--brand), var(--brand2)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.dp-lead { font-size:1.12rem; color:var(--muted); max-width:720px; margin:0 auto; line-height:1.6; }
.dp-cta { margin-top:22px; color:var(--muted); font-size:0.9rem; }
.dp-cta b { color:var(--text); }

/* stats strip */
.dp-stats { display:grid; grid-template-columns:repeat(4, 1fr); gap:14px; margin:28px 0 8px 0; }
.dp-stat { background:linear-gradient(180deg, rgba(24,33,74,0.85), rgba(18,25,53,0.85)); border:1px solid var(--line); border-radius:16px; padding:18px 20px; }
.dp-stat-n { font-size:2rem; font-weight:800; color:var(--text); }
.dp-stat-l { font-size:0.85rem; color:var(--muted); margin-top:2px; }

/* sections and cards */
.dp-section { margin:44px 0 16px 0; }
.dp-h2 { font-size:1.6rem; font-weight:800; color:var(--text); margin:0 0 6px 0; letter-spacing:-0.01em; }
.dp-sub { color:var(--muted); font-size:0.98rem; margin:0 0 18px 0; }
.dp-grid { display:grid; gap:14px; }
.dp-g2 { grid-template-columns:repeat(2, 1fr); } .dp-g3 { grid-template-columns:repeat(3, 1fr); } .dp-g4 { grid-template-columns:repeat(4, 1fr); }
.dp-card { background:rgba(18,25,53,0.85); border:1px solid var(--line); border-radius:16px; padding:18px 20px; height:100%;
           transition: transform .15s ease, border-color .15s ease; }
.dp-card:hover { transform:translateY(-2px); border-color:rgba(124,131,255,0.45); }
.dp-card h4 { margin:8px 0 6px 0; font-size:1.02rem; font-weight:700; color:var(--text); }
.dp-card p { margin:0; font-size:0.9rem; line-height:1.55; color:var(--muted); }
.dp-ico { width:38px; height:38px; border-radius:11px; display:grid; place-items:center; font-size:1.1rem;
          background:rgba(124,131,255,0.14); border:1px solid rgba(124,131,255,0.3); }
.dp-num { font-size:0.8rem; font-weight:700; color:var(--brand2); }
.dp-tag { display:inline-block; font-size:0.72rem; font-weight:600; padding:3px 9px; border-radius:999px; margin-top:10px; }
.dp-crew { color:#FDBA74; background:rgba(251,146,60,0.12); border:1px solid rgba(251,146,60,0.3); }
.dp-adk { color:#93C5FD; background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.3); }
.dp-tool { color:#FDE68A; background:rgba(250,204,21,0.10); border:1px solid rgba(250,204,21,0.28); }

/* KPI cards for results */
.dp-kpis { display:grid; grid-template-columns:repeat(6, 1fr); gap:12px; margin:8px 0 14px 0; }
.dp-kpi { background:rgba(18,25,53,0.9); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
.dp-kpi-l { font-size:0.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }
.dp-kpi-v { font-size:1.45rem; font-weight:800; color:var(--text); margin-top:4px; }
.dp-ok { color:var(--ok) !important; } .dp-bad { color:var(--bad) !important; }
.dp-banner { border-radius:14px; padding:14px 18px; margin:10px 0 4px 0; font-weight:600; }
.dp-banner.ok { background:rgba(52,211,153,0.10); border:1px solid rgba(52,211,153,0.35); color:#A7F3D0; }
.dp-banner.bad { background:rgba(248,113,113,0.10); border:1px solid rgba(248,113,113,0.35); color:#FECACA; }

/* pipeline */
.dp-flow { display:grid; grid-template-columns:repeat(7, 1fr); gap:8px; margin:14px 0 4px 0; }
.dp-node { position:relative; text-align:center; padding:12px 6px; border-radius:12px; border:1px solid var(--line); background:rgba(18,25,53,0.75); }
.dp-node b { display:block; font-size:0.86rem; color:var(--text); }
.dp-node span { font-size:0.72rem; color:var(--muted); }
.dp-node.done { border-color:rgba(52,211,153,0.5); background:rgba(52,211,153,0.07); }
.dp-node.wait { border-color:rgba(251,191,36,0.55); background:rgba(251,191,36,0.07); }

/* form widgets */
.stTextArea textarea, .stTextInput input { background:var(--panel) !important; border:1px solid var(--line) !important; border-radius:12px !important; color:var(--text) !important; }
[data-baseweb="select"] > div { background:var(--panel) !important; border:1px solid var(--line) !important; border-radius:12px !important; }
.stButton > button, .stDownloadButton > button { border-radius:12px; font-weight:600; border:1px solid var(--line); padding:0.55rem 1.1rem; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg, var(--brand), #4F7BFF 55%, var(--brand2)); border:none; color:#fff;
                                      box-shadow:0 8px 24px rgba(124,131,255,0.35); padding:0.7rem 1.6rem; font-size:1rem; }
.stButton > button[kind="primary"]:hover { filter:brightness(1.08); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px !important; border-color:var(--line) !important; background:rgba(18,25,53,0.6); }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:12px !important; background:rgba(18,25,53,0.6); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stWidgetLabel"] p { color:var(--muted) !important; font-weight:500 !important; }

/* footer */
.dp-footer { margin-top:56px; padding:22px 4px; border-top:1px solid var(--line); display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; color:var(--muted); font-size:0.85rem; }
.dp-footer b { color:var(--text); }

@media (max-width: 900px) {
  .dp-h1 { font-size:2.1rem; } .dp-stats, .dp-g4, .dp-kpis { grid-template-columns:repeat(2, 1fr); }
  .dp-g3, .dp-g2 { grid-template-columns:1fr; } .dp-flow { grid-template-columns:repeat(2, 1fr); }
}
</style>
""")

# ---------------------------------------------------------------------------
# Access gate (APP_PASSWORD secret). Skipped when unset, e.g. local development.
# ---------------------------------------------------------------------------
_APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
if _APP_PASSWORD and not st.session_state.get("authed"):
    html("""<div class="dp-hero"><div class="dp-logo" style="justify-content:center"><div class="dp-logo-mark">⚡</div>DevPilot AI</div>
            <div class="dp-h1" style="font-size:2.2rem">Sign in to the <span class="dp-grad">demo workspace</span></div></div>""")
    _c = st.columns([1, 1.2, 1])[1]
    _pw = _c.text_input("Access password", type="password")
    if _pw and _pw == _APP_PASSWORD:
        st.session_state["authed"] = True
        st.rerun()
    elif _pw:
        _c.error("Incorrect password.")
    st.stop()

# ---------------------------------------------------------------------------
# Data and constants
# ---------------------------------------------------------------------------
ARCH_OPTIONS = {
    "C1 · Hierarchical + Blackboard (proposed)": AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A,
    "C2 · Sequential pipeline": AgenticPatternType.SEQUENTIAL_BASELINE,
    "C3 · Peer-to-peer messaging": AgenticPatternType.HIERARCHICAL_P2P_A2A,
    "Critic-reflection loop": AgenticPatternType.CRITIC_REFLECTION_LOOP,
}
PERSONAS = {"Balanced Engineer": PersonaType.BALANCED_ENGINEER, "Strict Architect": PersonaType.STRICT_ARCHITECT,
            "Permissive Prototyper": PersonaType.PERMISSIVE_PROTOTYPER}
PROMPT_STYLES = {"Standard": "standard", "ReAct (tool-using)": "react", "Few-shot": "few_shot"}
PRESETS = {
    "T1 · Standard deviation": "Add a standard_deviation(numbers, sample=False) method to AdvancedCalculator in src/calculator.py. It must raise ValueError for an empty list, and for sample=True with fewer than two values.",
    "T2 · Median": "Add a median(numbers) method to AdvancedCalculator in src/calculator.py that returns the middle value (average of the two middle values for even-length input) and raises ValueError for an empty list.",
    "T3 · Safe percentage": "Add a safe_percentage(part, whole) method to AdvancedCalculator in src/calculator.py that returns part / whole * 100 rounded to 2 decimals and raises ValueError when whole is 0.",
    "T5 · Factorial": "Add a factorial(n) method to AdvancedCalculator in src/calculator.py that returns n! for non-negative integers and raises ValueError for negative numbers or non-integers.",
    "T6 · Modulo": "Add a modulo(a, b) method to AdvancedCalculator in src/calculator.py that returns a % b and raises ZeroDivisionError with the message 'Cannot take modulo by zero' when b is 0.",
    "T10 · Clamp": "Add a clamp(value, low, high) method to AdvancedCalculator in src/calculator.py that limits value to the range [low, high] and raises ValueError when low is greater than high.",
    "Custom requirement": "",
}
RESULTS_PATH = settings.SANDBOX_ROOT / "live_evaluation_results.json"
CONFIG_LABELS = {"C1": "C1 Hierarchical + Blackboard", "C2": "C2 Sequential", "C3": "C3 Peer-to-peer",
                 "C4": "C4 Cross-model (local coder)", "C1R": "C1 + ReAct prompt", "C1F": "C1 + Few-shot prompt",
                 "C1S": "C1 + Strict persona", "C1P": "C1 + Permissive persona"}
PALETTE = ["#7C83FF", "#22D3EE", "#F472B6", "#FBBF24", "#34D399"]

for key, default in (("metrics", None), ("judge", None), ("rag_run_id", None), ("compare_rows", []),
                     ("repo_url", "sample_repo"), ("repo_branch", "main"), ("rag_repo", None)):
    st.session_state.setdefault(key, default)


@st.cache_data(show_spinner=False)
def load_results():
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
    rows = [r for r in rows if r.get("arch") and not r.get("excluded_from_stats")]
    df = pd.DataFrame([{
        "Task": r["task_id"], "Config": r["arch"], "Success": bool(r.get("success")),
        "Tests": f"{r.get('tests_passed')}/{r.get('tests_total')}", "Repair cycles": r.get("recovery_cycles") or 0,
        "Time (s)": r.get("total_time_s"), "Tokens": r.get("total_tokens"), "Cost ($)": r.get("cost_usd"),
        "Coverage %": r.get("coverage_percent"), "Judge /5": (r.get("judge") or {}).get("overall"),
        "Retrieval P@5": (r.get("retrieval") or {}).get("precision_at_k"),
    } for r in rows])
    lat = {}
    for r in rows:
        if r["arch"] == "C1":
            for a in r.get("agents") or []:
                lat.setdefault(a["agent"].split(":")[-1].replace("Agent", ""), []).append(a.get("latency_s") or 0)
    lat_df = pd.DataFrame([{"Agent": k, "Latency (s)": sum(v) / len(v)} for k, v in lat.items()])
    return df, lat_df


def short_agent(agent_id: str) -> str:
    return agent_id.split(":")[-1].replace("Agent", "")


def is_remote(repo: str) -> bool:
    return repo.startswith(("https://", "http://"))


def repo_target(repo: str, branch: str):
    repo = (repo or "sample_repo").strip()
    if is_remote(repo):
        return RepositoryConfig(repo_url=repo, base_branch=(branch or "").strip() or None)
    return repo


def run_pipeline(prompt: str, pattern, persona, prompt_style, repo: str = "sample_repo", branch: str = "main"):
    global_blackboard.reset()
    tool_validation_stats.reset()
    orch = AutonomousEngineeringOrchestrator(
        pattern=pattern, persona=persona,
        coder_prompt_style=prompt_style if pattern == AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A else None,
    )
    return orch.execute_requirement(prompt, target_repo=repo_target(repo, branch))


def friendly_error(err: str) -> str:
    if "QUOTA" in err.upper() or "RESOURCE_EXHAUSTED" in err or "429" in err:
        return "The Gemini API daily quota for this key is used up. Update GEMINI_API_KEY in the app secrets and try again."
    if "503" in err or "UNAVAILABLE" in err:
        return "Gemini is temporarily overloaded (HTTP 503). Wait a minute and run again."
    return err


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, fmt: str = ",.1f", horizontal: bool = False, height: int = 260):
    base = alt.Chart(df).properties(height=height, title=alt.TitleParams(title, anchor="start", color="#E7EAF6",
                                                                        fontSize=14, font="Inter"))
    if horizontal:
        bars = base.mark_bar(cornerRadiusEnd=6, height=18).encode(
            y=alt.Y(f"{x}:N", sort="-x", title=None), x=alt.X(f"{y}:Q", title=None),
            color=alt.Color(f"{x}:N", legend=None, scale=alt.Scale(range=PALETTE)), tooltip=[x, alt.Tooltip(y, format=fmt)])
        text = base.mark_text(align="left", dx=5, color="#E7EAF6").encode(
            y=alt.Y(f"{x}:N", sort="-x"), x=f"{y}:Q", text=alt.Text(f"{y}:Q", format=fmt))
    else:
        bars = base.mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=46).encode(
            x=alt.X(f"{x}:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{y}:Q", title=None),
            color=alt.Color(f"{x}:N", legend=None, scale=alt.Scale(range=PALETTE)), tooltip=[x, alt.Tooltip(y, format=fmt)])
        text = base.mark_text(dy=-8, color="#E7EAF6", fontWeight=600).encode(
            x=alt.X(f"{x}:N", sort=None), y=f"{y}:Q", text=alt.Text(f"{y}:Q", format=fmt))
    chart = (bars + text).configure_view(strokeWidth=0).configure(background="transparent").configure_axis(
        labelColor="#9AA3C7", gridColor="rgba(148,163,255,0.10)", domainColor="rgba(148,163,255,0.2)",
        labelFont="Inter", tickColor="transparent")
    st.altair_chart(chart, width="stretch")


def kpis(items) -> None:
    cells = "".join(f'<div class="dp-kpi"><div class="dp-kpi-l">{esc(l)}</div><div class="dp-kpi-v {c}">{esc(v)}</div></div>'
                    for l, v, c in items)
    html(f'<div class="dp-kpis">{cells}</div>')


def section(title: str, sub: str = "") -> None:
    html(f'<div class="dp-section"><div class="dp-h2">{esc(title)}</div>'
         + (f'<div class="dp-sub">{esc(sub)}</div>' if sub else "") + "</div>")


results_df, latency_df = load_results()
main_df = results_df[results_df["Config"].isin(["C1", "C2", "C3", "C4"])] if not results_df.empty else results_df

# ---------------------------------------------------------------------------
# Navbar
# ---------------------------------------------------------------------------
html(f"""
<div class="dp-nav">
  <div class="dp-logo"><div class="dp-logo-mark">⚡</div>DevPilot AI</div>
  <div class="dp-nav-right">
    <span class="dp-pill dp-live">{'Live' if settings.EXECUTION_MODE == 'live' else 'Simulation'}</span>
    <span class="dp-pill">{esc(settings.DEFAULT_MODEL)}</span>
    <span class="dp-pill">CrewAI + Google ADK</span>
  </div>
</div>
""")

tab_home, tab_run, tab_arch, tab_eval, tab_code = st.tabs(["Home", "Studio", "Architecture", "Results", "Codebase"])

# =============================================================================
# HOME
# =============================================================================
with tab_home:
    c1_rate = f"{int(main_df[main_df['Config'] == 'C1']['Success'].mean() * 100)}%" if not main_df.empty else "-"
    avg_cost = f"${results_df['Cost ($)'].mean():.4f}" if not results_df.empty else "-"
    html(f"""
    <div class="dp-hero">
      <span class="dp-eyebrow">Multi-agent software engineering</span>
      <div class="dp-h1">From a plain-English requirement to a <span class="dp-grad">tested, reviewed code change</span></div>
      <div class="dp-lead">DevPilot AI is a team of seven AI agents built with CrewAI and Google ADK. They read the codebase,
      plan the change, write the code and its tests, diagnose their own failures and re-plan, then wait for a human to approve.</div>
      <div class="dp-cta">Open the <b>Studio</b> tab to run the agents on a live repository.</div>
    </div>
    <div class="dp-stats">
      <div class="dp-stat"><div class="dp-stat-n">{len(results_df)}</div><div class="dp-stat-l">live evaluation runs</div></div>
      <div class="dp-stat"><div class="dp-stat-n">{c1_rate}</div><div class="dp-stat-l">success rate of C1 (10 tasks)</div></div>
      <div class="dp-stat"><div class="dp-stat-n">7</div><div class="dp-stat-l">agents across two frameworks</div></div>
      <div class="dp-stat"><div class="dp-stat-n">{avg_cost}</div><div class="dp-stat-l">average cost per run</div></div>
    </div>
    """)

    section("How it works", "Plan, act, observe, reason, re-plan: the loop that makes the system reliable.")
    steps = [("01", "🔎", "Understand", "Guardrails screen the request, then the RAG agent retrieves the relevant code with AST-aware search."),
             ("02", "🧭", "Plan", "The CrewAI Planner writes acceptance criteria and an ordered plan onto the shared Blackboard."),
             ("03", "🛠️", "Build & test", "The ADK Coder edits the code in a sandbox; the Tester writes new tests and runs them with the old suite through MCP."),
             ("04", "♻️", "Repair & approve", "On failure the Debugger diagnoses, the Manager re-plans (up to 3 cycles). A human approves the final diff.")]
    html('<div class="dp-grid dp-g4">' + "".join(
        f'<div class="dp-card"><div class="dp-ico">{i}</div><div class="dp-num" style="margin-top:12px">STEP {n}</div><h4>{t}</h4><p>{d}</p></div>'
        for n, i, t, d in steps) + "</div>")

    section("The agent team", "CrewAI handles management and review; Google ADK handles the technical work.")
    agents = [("🧭", "Planner", "Turns the requirement into acceptance criteria and a step-by-step plan.", "crew", "CrewAI"),
              ("🎯", "Manager", "Reads the Debugger's diagnosis and decides: fetch context, retry fix, or stop.", "crew", "CrewAI"),
              ("🛡️", "Auditor", "Security review: AST scan for dangerous calls before anything is delivered.", "crew", "CrewAI"),
              ("📚", "RAG", "Indexes the repository into code chunks and retrieves what the change touches.", "adk", "Google ADK"),
              ("💻", "Coder", "Writes the change. Supports standard, ReAct (tool-using) and few-shot prompting.", "adk", "Google ADK"),
              ("🧪", "Tester", "Generates new tests and runs the full suite, checking for regressions.", "adk", "Google ADK"),
              ("🩺", "Debugger", "Classifies why tests fail and proposes a targeted fix.", "adk", "Google ADK"),
              ("⚖️", "Judge", "A different model scores plan, code and tests from 1 to 5.", "adk", "Google ADK (evaluation)"),
              ("🔧", "MCP tools", "repo_tool and test_runner servers: sandbox, patch, diff, pytest, coverage.", "tool", "MCP")]
    html('<div class="dp-grid dp-g3">' + "".join(
        f'<div class="dp-card"><div class="dp-ico">{i}</div><h4>{n}</h4><p>{d}</p><span class="dp-tag dp-{c}">{t}</span></div>'
        for i, n, d, c, t in agents) + "</div>")

    section("What makes it different")
    feats = [("♻️", "Failure-driven re-planning", "Failures are classified, not just retried. The Manager picks a different remedy for each kind of failure."),
             ("🧱", "Blackboard + A2A", "Agents share one versioned state and exchange signed A2A messages across CrewAI and ADK."),
             ("🔐", "Secure by design", "Prompt-injection and toxicity guardrails, AST scanning, sandboxing and a human approval gate."),
             ("📈", "Measured honestly", "Every run records per-agent tokens, latency and cost from the provider. Quota failures never count as success.")]
    html('<div class="dp-grid dp-g2">' + "".join(
        f'<div class="dp-card"><div class="dp-ico">{i}</div><h4>{t}</h4><p>{d}</p></div>' for i, t, d in feats) + "</div>")

# =============================================================================
# STUDIO
# =============================================================================
with tab_run:
    section("Engineering Studio", "Point the agents at a repository, describe a change, and let them build and test it.")
    with st.container(border=True):
        r1, r2 = st.columns([4, 1])
        st.session_state.repo_url = r1.text_input(
            "Repository", value=st.session_state.repo_url, key="inp_repo",
            help="'sample_repo' (built-in calculator project) or any public GitHub HTTPS URL of a Python project with pytest tests.")
        st.session_state.repo_branch = r2.text_input("Branch", value=st.session_state.repo_branch, key="inp_branch")
        if is_remote(st.session_state.repo_url):
            st.caption("The repository is cloned into an isolated sandbox through the MCP repo_tool. Works best with small, "
                       "pure-Python projects that have a pytest suite and no heavy dependencies. Write a custom requirement "
                       "naming the file to change.")
        left, right = st.columns([3, 2], gap="large")
        with left:
            preset = st.selectbox("Example task", list(PRESETS), index=0, key="sel_preset")
            prompt = st.text_area("Requirement", value=PRESETS[preset], height=128,
                                  placeholder="e.g. Add a cube(x) method to AdvancedCalculator that returns x ** 3.")
        with right:
            arch_label = st.selectbox("Architecture", list(ARCH_OPTIONS), key="sel_arch")
            c_a, c_b = st.columns(2)
            persona_label = c_a.selectbox("Persona", list(PERSONAS), key="sel_persona")
            style_label = c_b.selectbox("Coder prompt", list(PROMPT_STYLES), key="sel_style",
                                        disabled=ARCH_OPTIONS[arch_label] != AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A)
            st.caption("C4 uses a local Ollama model, so it runs only on a laptop; its results are under Results.")
            run_clicked = st.button("⚡  Run the agents", type="primary", key="btn_run_pipeline",
                                    disabled=not prompt.strip(), width="stretch")

    if run_clicked:
        st.session_state.judge = None
        with st.status("Agents at work: retrieving code, planning, coding, testing…", expanded=False) as status:
            try:
                st.session_state.metrics = run_pipeline(prompt.strip(), ARCH_OPTIONS[arch_label],
                                                        PERSONAS[persona_label], PROMPT_STYLES[style_label],
                                                        st.session_state.repo_url, st.session_state.repo_branch)
                status.update(label="Run complete", state="complete" if st.session_state.metrics.success else "error")
            except Exception as exc:  # noqa: BLE001
                st.session_state.metrics = None
                status.update(label="Run failed", state="error")
                st.error(friendly_error(f"{type(exc).__name__}: {exc}"))

    m = st.session_state.metrics
    done = m is not None
    flow = [("RAG", "ADK"), ("Plan", "CrewAI"), ("Code", "ADK"), ("Test", "ADK + MCP"),
            ("Repair", "Debugger · Manager"), ("Audit", "CrewAI"), ("Approve", "Human")]
    def node_state(step: str):
        """(css class, prefix, subtitle override) for one pipeline step after a run."""
        if not done:
            return "", "", None
        if step == "Repair":
            n = m.failure_recovery_cycles
            return ("done", "✓ ", f"{n} cycle{'s' if n != 1 else ''}") if n else ("", "", "not needed")
        if step == "Approve":
            status_ = global_blackboard.state.human_approval_status
            if status_ == "approved":
                return "done", "✓ ", "approved"
            return ("", "", "rejected") if status_ == "rejected" else ("wait", "", "waiting for you")
        return ("done", "✓ ", None) if m.success or step in ("RAG", "Plan", "Code", "Test") else ("", "", None)

    cells = []
    for t, s in flow:
        cls, pre, sub = node_state(t)
        cells.append(f'<div class="dp-node {cls}"><b>{pre}{t}</b><span>{sub or s}</span></div>')
    html('<div class="dp-flow">' + "".join(cells) + "</div>")

    if m is None:
        html('<div class="dp-card" style="text-align:center;padding:34px;margin-top:14px"><div class="dp-ico" style="margin:0 auto">🤖</div>'
             '<h4>Ready when you are</h4><p>Pick a task and press <b>Run the agents</b>. A run takes 30-60 seconds and makes 3-5 Gemini calls.</p></div>')
    else:
        bb = global_blackboard.state
        if m.success:
            html('<div class="dp-banner ok">✅ All tests pass, no regressions and the security audit is clean. Review the change and approve it below.</div>')
        else:
            html(f'<div class="dp-banner bad">❌ The run did not pass. {esc(friendly_error(m.error_details or ""))[:400]}</div>')
        cov = (m.quality or {}).get("coverage") or {}
        kpis([("Result", "Passed" if m.success else "Failed", "dp-ok" if m.success else "dp-bad"),
              ("Tests", f"{m.tests_passed}/{m.tests_total}", ""), ("Time", f"{m.total_execution_time_seconds:.1f}s", ""),
              ("Tokens", f"{m.total_tokens:,}", ""), ("Cost", f"${m.total_cost_usd:.5f}", ""),
              ("Coverage", f"{cov.get('percent_covered', '-')}%", "")])
        st.caption(f"Run {m.run_id} · {m.architecture_pattern} · {m.persona} · {m.failure_recovery_cycles} repair cycles · "
                   f"{m.a2a_message_count} A2A messages · {m.tool_call_count} tool calls")

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
                st.markdown(f"**{rep.passed} of {rep.total_tests} tests passed** in {rep.execution_time_seconds:.2f}s "
                            "(existing suite + tests written by the ADK Tester).")
                st.dataframe(pd.DataFrame([{"Test": i.test_node_id, "Result": "✅ passed" if i.outcome == "passed" else f"❌ {i.outcome}",
                                            "Message": (i.failure_message or "")[:200]} for i in rep.items]),
                             hide_index=True, width="stretch")
            regs = bb.metadata.get("regressions") or []
            snap = tool_validation_stats.snapshot()
            st.caption(f"Regressions vs. original code: {len(regs) or 'none'} · Tool-output validation: "
                       f"{snap['checked']} MCP results type-checked, {len(snap['violations'])} violations")
        with t_agents:
            rows = [{"Agent": short_agent(b.agent_id), "Framework": b.framework, "Latency (s)": round(b.latency_seconds, 2),
                     "LLM calls": b.llm_calls, "Tokens": b.total_tokens, "Cost ($)": round(b.estimated_cost_usd, 6)}
                    for b in m.agent_breakdowns]
            if rows:
                adf = pd.DataFrame(rows)
                c1, c2 = st.columns([3, 2])
                c1.dataframe(adf, hide_index=True, width="stretch")
                with c2:
                    bar_chart(adf, "Agent", "Latency (s)", "Response time per agent (s)", fmt=".2f", horizontal=True, height=230)
            if st.button("Score this run with the LLM judge (1 extra call)", key="btn_judge"):
                try:
                    from agents.google_adk.judge_agent import GoogleADKJudgeAgent
                    with st.spinner("The judge (gemini-3.5-flash-lite) is scoring the run…"):
                        st.session_state.judge = GoogleADKJudgeAgent().judge()
                except Exception as exc:  # noqa: BLE001
                    st.error(friendly_error(str(exc)))
            v = st.session_state.judge
            if v is not None:
                crit = [("Plan quality", v.plan_quality), ("Code correctness", v.code_correctness),
                        ("Code quality", v.code_quality), ("Test adequacy", v.test_adequacy)]
                if v.diagnosis_quality:
                    crit.append(("Diagnosis quality", v.diagnosis_quality))
                kpis([("Judge overall", f"{v.overall:.2f} / 5", "dp-ok")] + [(n, f"{c.score}/5", "") for n, c in crit][:5])
                st.dataframe(pd.DataFrame([{"Criterion": n, "Score": c.score, "Why": c.justification} for n, c in crit]),
                             hide_index=True, width="stretch")
        with t_plan:
            if bb.analysis:
                st.markdown(f"**Requirement understood as:** {bb.analysis.title}. {bb.analysis.problem_statement}")
                for c in bb.analysis.acceptance_criteria:
                    st.markdown(f"- `{c.id}` {c.description}")
            if bb.plan:
                st.dataframe(pd.DataFrame([{"Step": t.title, "Framework": t.assigned_framework, "Role": t.assigned_role,
                                            "Status": t.status.value} for t in bb.plan.tasks]), hide_index=True, width="stretch")
            if bb.failure_diagnoses or bb.manager_decisions:
                for d in bb.failure_diagnoses:
                    st.warning(f"Cycle {d.cycle} · Debugger diagnosis: **{d.category.value}**: {d.root_cause}")
                for d in bb.manager_decisions:
                    st.info(f"Cycle {d.cycle} · Manager decision: **{d.action.value}**: {d.rationale}")
            else:
                st.caption("No failures this run, so no re-planning was needed.")
            if bb.security_audit:
                st.caption(f"Security audit (CrewAI Auditor): {'passed' if bb.security_audit.passed else 'issues found'} · "
                           f"Blackboard version v{bb.version}")
        with t_msgs:
            hist = global_a2a_bus.get_history()
            if hist:
                st.dataframe(pd.DataFrame([{"Time": e.timestamp.strftime("%H:%M:%S"), "From": e.sender_id, "To": e.recipient_id,
                                            "Type": e.message_type.value, "Signature": e.signature[:12] + "…"} for e in hist]),
                             hide_index=True, width="stretch")
            else:
                st.caption("No A2A messages recorded.")
        with t_approve:
            st.markdown(f"Approval status: **{bb.human_approval_status.upper()}**")
            st.caption("Nothing is committed until a human approves. Approval commits the change on an `agentic/*` branch of the sandbox.")
            c1, c2 = st.columns(2)
            notes = c1.text_input("Reviewer notes", value="Tests pass and the diff is minimal.")
            if c1.button("✅ Approve", key="btn_hitl_approve", disabled=not m.success):
                res = hitl_gatekeeper.approve_and_commit(run_id=m.run_id, approver="Reviewer", notes=notes,
                                                         target_repo=st.session_state.repo_url)
                (st.success if res.get("success") else st.error)(res.get("message") or res.get("error") or "Done")
            reason = c2.text_input("Rejection reason", value="Does not meet the requirement.")
            if c2.button("❌ Reject", key="btn_hitl_reject"):
                hitl_gatekeeper.reject(reason)
                st.warning("Change rejected; nothing was committed.")

# =============================================================================
# ARCHITECTURE
# =============================================================================
with tab_arch:
    section("System architecture", "Hierarchical multi-agent system (C1): CrewAI manages, Google ADK executes, MCP provides the tools.")
    with st.container(border=True):
        st.graphviz_chart(
            """
            digraph G {
              rankdir=LR; bgcolor="transparent"; pad=0.3; nodesep=0.35; ranksep=0.55;
              node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#2B3566", fontcolor="#E7EAF6", fillcolor="#18214A", penwidth=1.2];
              edge [fontname="Helvetica", fontsize=9, color="#7C83FF", fontcolor="#9AA3C7", penwidth=1.1];
              U [label="Developer\\nrequirement", fillcolor="#232C5C"];
              G [label="Guardrails\\ninjection · toxicity", fillcolor="#4A1F2B", color="#F87171"];
              subgraph cluster_crew { label="CrewAI"; fontcolor="#FDBA74"; style="rounded,dashed"; color="#FB923C";
                P [label="Planner"]; M [label="Manager"]; A [label="Auditor"]; }
              subgraph cluster_adk { label="Google ADK"; fontcolor="#93C5FD"; style="rounded,dashed"; color="#3B82F6";
                R [label="RAG"]; C [label="Coder"]; T [label="Tester"]; D [label="Debugger"]; }
              B [label="Blackboard\\nshared versioned state", shape=cylinder, fillcolor="#3A3313", color="#FBBF24"];
              subgraph cluster_mcp { label="MCP servers"; fontcolor="#FDE68A"; style="rounded,dashed"; color="#CA8A04";
                RT [label="repo_tool\\nsandbox · read · patch · diff · RAG", fillcolor="#2A2610"];
                TR [label="test_runner\\nsyntax · pytest · coverage", fillcolor="#2A2610"]; }
              H [label="Human\\napproval", fillcolor="#123B2C", color="#34D399"];
              U -> G -> R -> P -> C -> T;
              T -> A [label="pass"]; A -> H;
              T -> D [label="fail", color="#F87171"]; D -> M [label="diagnosis", color="#F87171"];
              M -> P [label="re-plan", color="#F87171"]; M -> C [label="retry fix", color="#F87171"];
              {R C T} -> RT [style=dotted, arrowhead=none, color="#CA8A04"]; T -> TR [style=dotted, arrowhead=none, color="#CA8A04"];
              {P C T D M} -> B [style=dashed, arrowhead=none, color="#FBBF24"];
            }
            """,
            width="stretch",
        )
    st.caption("Agents exchange SHA-256 signed A2A envelopes across CrewAI and ADK and share state on the Blackboard. "
               "Tools are reached only through MCP, and every tool result is type-checked by a validation callback.")

    section("Configurations compared", "Each configuration ran the same 10 tasks with live Gemini calls.")
    confs = [("C1", "Hierarchical + Blackboard", "Manager-led team on a shared Blackboard with A2A messages. Debugger diagnoses failures and the Manager re-plans.", "Proposed"),
             ("C2", "Sequential pipeline", "Same agents and tools in a fixed order, no Manager and no repair loop.", "Baseline"),
             ("C3", "Peer-to-peer messaging", "Same roles with no Blackboard and no Manager hub: each agent only sees the messages sent to it.", "Messaging comparison"),
             ("C4", "Cross-model", "C1 with a local Ollama coder (qwen2.5:1.5b) and learned embeddings (nomic-embed-text) for RAG.", "Model comparison")]
    html('<div class="dp-grid dp-g4">' + "".join(
        f'<div class="dp-card"><div class="dp-num">{c}</div><h4>{t}</h4><p>{d}</p><span class="dp-tag dp-adk">{tag}</span></div>'
        for c, t, d, tag in confs) + "</div>")
    html('<div class="dp-card" style="margin-top:14px"><h4>Why C1 was chosen</h4><p>It is the only design that classifies a failure '
         'and picks a different remedy (fetch more context, fix the code, or stop). In the live runs C1 succeeded 10/10; the sequential '
         'C2 was cheapest but failed 2/10 because it cannot repair; C3 matched C1 on these tasks but has no global view for '
         're-planning decisions.</p></div>')

    section("Security and observability")
    sec = [("🛡️", "Guardrails", "Prompt-injection and toxic-language checks on input; AST scan of generated code for eval, os.system, subprocess."),
           ("🧰", "Safeguards", "Per-run sandbox, path guard, retry limit and no-progress stop, human approval before commit, pushes only to agentic/*."),
           ("📡", "Callbacks", "Agent start/finish/error, tool start/end and guardrail callbacks; every MCP result is schema-validated."),
           ("🧾", "Provenance", "Usage ledger with per-agent tokens, latency and cost; signed A2A envelopes and a SHA-256 chained provenance log.")]
    html('<div class="dp-grid dp-g4">' + "".join(
        f'<div class="dp-card"><div class="dp-ico">{i}</div><h4>{t}</h4><p>{d}</p></div>' for i, t, d in sec) + "</div>")

# =============================================================================
# RESULTS
# =============================================================================
with tab_eval:
    section("Evaluation results", "10 tasks × 4 configurations, plus prompt and persona variants: every run made real Gemini calls.")
    if results_df.empty:
        st.info("No live evaluation results found.")
    else:
        kpis([("Live runs", len(results_df), ""), ("Success C1-C4", f"{int(main_df['Success'].sum())}/{len(main_df)}", "dp-ok"),
              ("Avg time", f"{results_df['Time (s)'].mean():.1f}s", ""), ("Avg tokens", f"{results_df['Tokens'].mean():,.0f}", ""),
              ("Avg cost", f"${results_df['Cost ($)'].mean():.5f}", ""), ("Judge score", f"{results_df['Judge /5'].mean():.2f}/5", "")])

        def summarize(frame, configs):
            g = frame[frame["Config"].isin(configs)].groupby("Config")
            out = pd.DataFrame({"Runs": g.size(), "Success %": (g["Success"].mean() * 100).round(0),
                                "Avg time (s)": g["Time (s)"].mean().round(1), "Avg tokens": g["Tokens"].mean().round(0),
                                "Avg cost ($)": g["Cost ($)"].mean().round(5), "Repair cycles": g["Repair cycles"].mean().round(2),
                                "Coverage %": g["Coverage %"].mean().round(1), "Judge /5": g["Judge /5"].mean().round(2),
                                }).reindex([c for c in configs if c in g.groups])
            out.insert(0, "Short", out.index)
            out.insert(1, "Configuration", [CONFIG_LABELS.get(c, c) for c in out.index])
            return out.reset_index(drop=True)

        arch = summarize(main_df, ["C1", "C2", "C3", "C4"])
        with st.container(border=True):
            g1, g2, g3 = st.columns(3)
            with g1:
                bar_chart(arch, "Short", "Success %", "Success rate (%)", fmt=".0f")
            with g2:
                bar_chart(arch, "Short", "Avg time (s)", "Avg time per run (s)")
            with g3:
                bar_chart(arch, "Short", "Avg tokens", "Avg tokens per run", fmt=",.0f")
        st.dataframe(arch.drop(columns=["Short"]), hide_index=True, width="stretch")
        html('<div class="dp-grid dp-g3" style="margin-top:12px">'
             '<div class="dp-card"><h4>The repair loop matters</h4><p>C2 failed T7 and T8: its generated tests failed and it has no way to repair. C1, C3 and C4 succeeded 10/10.</p></div>'
             '<div class="dp-card"><h4>Cheaper is not better</h4><p>C2 used less than half the tokens of C1, at the cost of reliability and lower test coverage.</p></div>'
             '<div class="dp-card"><h4>Model choice per role</h4><p>C4\'s small local coder needed repair cycles more often but always recovered; its learned embeddings retrieved the best context.</p></div>'
             '</div>')

        section("Prompt and persona variance", "Same architecture (C1) and tasks T1-T2; only the Coder prompt or the team persona changes.")
        var = summarize(results_df[results_df["Task"].isin(["T1", "T2"])], ["C1", "C1R", "C1F", "C1S", "C1P"])
        var["Short"] = ["Standard" if c == "C1" else CONFIG_LABELS[c].replace("C1 + ", "") for c in var["Short"]]
        st.dataframe(var.drop(columns=["Short"]), hide_index=True, width="stretch")
        with st.container(border=True):
            v1, v2 = st.columns(2)
            with v1:
                bar_chart(var, "Short", "Avg tokens", "Avg tokens per run", fmt=",.0f", horizontal=True, height=210)
            with v2:
                bar_chart(var, "Short", "Avg time (s)", "Avg time per run (s)", horizontal=True, height=210)

        if not latency_df.empty:
            section("Response time per agent", "Average over the 10 C1 runs.")
            with st.container(border=True):
                bar_chart(latency_df, "Agent", "Latency (s)", "Average latency (s)", fmt=".2f", horizontal=True, height=280)

        with st.expander("All runs"):
            st.dataframe(results_df, hide_index=True, width="stretch")
            st.download_button("Download results (JSON)", RESULTS_PATH.read_bytes(), "live_evaluation_results.json")

    section("Run a live comparison", "Sends one requirement through C1, C2 and C3 now (about 10-15 Gemini calls).")
    cmp_prompt = st.text_input("Requirement", value=PRESETS["T3 · Safe percentage"], key="cmp_prompt")
    if st.button("Run C1, C2 and C3", key="btn_run_compare"):
        st.session_state.compare_rows = []
        for label in list(ARCH_OPTIONS)[:3]:
            with st.spinner(f"Running {label}…"):
                try:
                    mm = run_pipeline(cmp_prompt, ARCH_OPTIONS[label], PersonaType.BALANCED_ENGINEER, "standard")
                    st.session_state.compare_rows.append({
                        "Configuration": label, "Success": "✅" if mm.success else "❌", "Tests": f"{mm.tests_passed}/{mm.tests_total}",
                        "Time (s)": round(mm.total_execution_time_seconds, 1), "Tokens": mm.total_tokens,
                        "Cost ($)": round(mm.total_cost_usd, 5), "Repair cycles": mm.failure_recovery_cycles})
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{label}: {friendly_error(str(exc))}")
                    break
    if st.session_state.compare_rows:
        st.dataframe(pd.DataFrame(st.session_state.compare_rows), hide_index=True, width="stretch")

# =============================================================================
# CODEBASE
# =============================================================================
with tab_code:
    section("Codebase", "Browse and search the selected repository. Change it in the Studio tab; every run works on an isolated sandbox copy.")
    cur_repo = (st.session_state.repo_url or "sample_repo").strip()
    st.markdown(f"Repository: `{cur_repo}`" + (f" · branch `{st.session_state.repo_branch}`" if is_remote(cur_repo) else ""))
    reindex = st.button("Re-index repository", key="btn_connect_repo")
    if st.session_state.rag_run_id is None or reindex or st.session_state.rag_repo != cur_repo:
        with st.spinner("Cloning and indexing the repository (AST chunks → vector store)…"):
            sb = (mcp_repo_client.clone_repository(cur_repo, base_branch=(st.session_state.repo_branch or "").strip() or None)
                  if is_remote(cur_repo) else mcp_repo_client.create_sandbox(cur_repo))
            st.session_state.rag_repo = cur_repo
            if sb.get("success"):
                st.session_state.rag_run_id = sb["run_id"]
                st.session_state.rag_stats = mcp_repo_client.index_rag(sb["run_id"])
            else:
                st.session_state.rag_run_id, st.session_state.rag_stats = None, {}
                st.error(f"Could not load the repository: {sb.get('error')}")
    stats = st.session_state.get("rag_stats") or {}
    kpis([("Index", "Ready" if stats.get("success") else "Not indexed", "dp-ok" if stats.get("success") else "dp-bad"),
          ("Code chunks", stats.get("total_chunks_scanned", 0), ""), ("Chunking", "Python AST", ""),
          ("Embedding", "Feature hashing", ""), ("Vector store", "ChromaDB", ""), ("Access", "MCP repo_tool", "")])

    left, right = st.columns([2, 3], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Search the code (RAG)**")
            q = st.text_input("Question", value="where is the average of a list computed?", key="rag_q")
            top_k = st.slider("Results", 1, 8, 4)
            if st.button("Search", key="btn_rag_search", type="primary"):
                res = mcp_repo_client.query_rag(query=q, run_id=st.session_state.rag_run_id, top_k=top_k)
                chunks = res.get("results", [])
                if not chunks:
                    st.warning("No matching code found.")
                for i, ch in enumerate(chunks):
                    with st.expander(f"{i + 1}. {ch.get('symbol_name')} · {ch.get('file_path')} "
                                     f"(lines {ch.get('start_line')}-{ch.get('end_line')})", expanded=i == 0):
                        st.code(ch.get("content", ""), language="python")
    with right:
        with st.container(border=True):
            st.markdown("**Browse files**")
            files = []
            if st.session_state.rag_run_id:
                files = [f for f in (mcp_repo_client.list_files(st.session_state.rag_run_id).get("files") or [])
                         if str(f).endswith((".py", ".toml", ".md", ".txt"))]
            if files:
                sel = st.selectbox("File", files, index=files.index("src/calculator.py") if "src/calculator.py" in files else 0,
                                   key="sel_file")
                st.code(mcp_repo_client.read_file(st.session_state.rag_run_id, sel).get("content", ""),
                        language="python" if sel.endswith(".py") else None)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
html("""
<div class="dp-footer">
  <div><b>DevPilot AI</b> · Multi-Agent System for Autonomous Software Engineering</div>
  <div>Muskan Patni (230534) · Vaishnavi Singh (230508) · Guide: Prof. Pranshu Tiwari · BML Munjal University</div>
</div>
""")
