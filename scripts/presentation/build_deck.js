// Builds the 15-slide (Slide 0-14) final presentation for DevPilot AI.
// Usage: node build_deck.js <path-to-live_evaluation_results.json> <output.pptx>
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const RESULTS_PATH = process.argv[2];
const OUT = process.argv[3] || "DevPilot_AI_Final_Presentation.pptx";
let results = [];
try { results = JSON.parse(fs.readFileSync(RESULTS_PATH, "utf-8")); } catch (e) { results = []; }

// ---------- palette & fonts ----------
const NAVY = "0F2A3D", TEAL = "13A89E", AMBER = "F2A541", INK = "1B2430", MUTED = "5B6B7A",
  LIGHT = "EEF4F6", WHITE = "FFFFFF", CORAL = "E76F51", SLATE = "3D5A80";
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Muskan Patni, Vaishnavi Singh";
pres.title = "DevPilot AI - Multi-Agent Autonomous Software Engineering";

pres.defineSlideMaster({
  title: "DARK",
  background: { color: NAVY },
  objects: [],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: WHITE },
  objects: [
    { text: { text: "DevPilot AI  |  BML Munjal University", options: { x: 0.5, y: 5.25, w: 6, h: 0.25, fontFace: BODY, fontSize: 9, color: MUTED, isTextBox: true, margin: 0 } } },
  ],
  slideNumber: { x: 9.2, y: 5.22, w: 0.4, h: 0.25, fontFace: BODY, fontSize: 9, color: MUTED },
});

function title(slide, text, sub) {
  slide.addText(text, { x: 0.5, y: 0.3, w: 9, h: 0.6, fontFace: HEAD, fontSize: 26, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  if (sub) slide.addText(sub, { x: 0.5, y: 0.88, w: 9, h: 0.3, fontFace: BODY, fontSize: 12, italic: true, color: MUTED, isTextBox: true, margin: 0 });
}
function card(slide, x, y, w, h, head, body, color, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: opts.fill || LIGHT }, line: { color: opts.fill || LIGHT } });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y + 0.15, w: 0.32, h: 0.32, fill: { color }, line: { color } });
  if (opts.badge) slide.addText(opts.badge, { x: x + 0.15, y: y + 0.15, w: 0.32, h: 0.32, fontFace: BODY, fontSize: 10, bold: true, color: WHITE, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addText(head, { x: x + 0.55, y: y + 0.12, w: w - 0.65, h: 0.38, fontFace: BODY, fontSize: opts.headSize || 13, bold: true, color: INK, valign: "middle", isTextBox: true, margin: 0 });
  if (body) slide.addText(body, { x: x + 0.15, y: y + 0.55, w: w - 0.3, h: h - 0.65, fontFace: BODY, fontSize: opts.bodySize || 11, color: INK, valign: "top", isTextBox: true, margin: 0, paraSpaceAfter: 3 });
}
function box(slide, x, y, w, h, text, fill, color = WHITE, size = 11, bold = true) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: fill } });
  slide.addText(text, { x, y, w, h, fontFace: BODY, fontSize: size, bold, color, align: "center", valign: "middle", isTextBox: true, margin: 2 });
}
function arrow(slide, x1, y1, x2, y2, color = MUTED) {
  slide.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 1.5, endArrowType: "triangle" } });
}
function bullets(items, size = 13) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, fontSize: size } }));
}

// ---------- aggregate live results ----------
const ARCH_NAMES = { C1: "C1 Hierarchical + Blackboard", C2: "C2 Sequential pipeline", C3: "C3 Peer-to-peer (no Blackboard)", C4: "C4 Cross-model (Ollama coder)" };
const byArch = {};
for (const r of results) {
  if (!r.arch || r.excluded_from_stats) continue;
  (byArch[r.arch] = byArch[r.arch] || []).push(r);
}
const archRows = ["C1", "C2", "C3", "C4"].filter(a => byArch[a]).map(a => {
  const rs = byArch[a];
  const ok = rs.filter(r => r.success).length;
  const done = rs.filter(r => r.run_id);
  const avg = (k) => done.length ? done.reduce((s, r) => s + (Number(r[k]) || 0), 0) / done.length : 0;
  const judged = done.filter(r => r.judge && r.judge.overall);
  const withRet = done.filter(r => r.retrieval);
  const withCov = done.filter(r => r.coverage_percent != null);
  return { a, runs: rs.length, ok, rate: rs.length ? Math.round(100 * ok / rs.length) : 0,
    time: avg("total_time_s"), tokens: Math.round(avg("total_tokens")), cycles: avg("recovery_cycles"),
    cost: avg("cost_usd"),
    cov: withCov.length ? withCov.reduce((t, r) => t + r.coverage_percent, 0) / withCov.length : null,
    judge: judged.length ? judged.reduce((t, r) => t + r.judge.overall, 0) / judged.length : null,
    prec: withRet.length ? withRet.reduce((t, r) => t + r.retrieval.precision_at_k, 0) / withRet.length : null };
});
const haveResults = archRows.length > 0;
// per-agent latency from the C1 runs
const agentLat = {};
for (const r of (byArch.C1 || [])) for (const ag of (r.agents || [])) {
  const k = ag.agent.replace("GoogleADK:", "ADK ").replace("CrewAI:", "CrewAI ").replace("MCP:", "MCP ");
  (agentLat[k] = agentLat[k] || []).push(Number(ag.latency_s) || 0);
}
const agentNames = Object.keys(agentLat);
const agentAvg = agentNames.map(k => +(agentLat[k].reduce((a, b) => a + b, 0) / agentLat[k].length).toFixed(2));
const totalRuns = results.filter(r => r.arch && !r.excluded_from_stats).length;
const nTasks = new Set(results.filter(r => r.arch).map(r => r.task_id)).size;
const fmt = (v, d = 1) => (v == null ? "-" : Number(v).toFixed(d));

// ======================= Slide 0: Title =======================
let s = pres.addSlide({ masterName: "DARK" });
s.addText("Final Project Presentation", { x: 0.6, y: 0.55, w: 8.8, h: 0.35, fontFace: BODY, fontSize: 14, color: AMBER, bold: true, isTextBox: true, margin: 0 });
s.addText("Multi-Agent AI System for Autonomous Software Engineering", { x: 0.6, y: 0.95, w: 8.8, h: 1.1, fontFace: HEAD, fontSize: 32, bold: true, color: WHITE, isTextBox: true, margin: 0 });
s.addText("Converting developer requirements into validated code changes  -  DevPilot AI", { x: 0.6, y: 2.1, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 15, italic: true, color: "CADCE6", isTextBox: true, margin: 0 });
box(s, 0.6, 2.95, 3.9, 0.75, "Muskan Patni  -  230534", TEAL, WHITE, 15);
box(s, 4.7, 2.95, 3.9, 0.75, "Vaishnavi Singh  -  230508", TEAL, WHITE, 15);
s.addText([
  { text: "Project Guide: Prof. Pranshu Tiwari", options: { breakLine: true } },
  { text: "BML Munjal University" },
], { x: 0.6, y: 4.05, w: 8.8, h: 0.7, fontFace: BODY, fontSize: 13, color: "CADCE6", isTextBox: true, margin: 0 });
s.addText("CrewAI  +  Google ADK  +  MCP  +  A2A  +  Blackboard", { x: 0.6, y: 4.85, w: 8.8, h: 0.3, fontFace: BODY, fontSize: 11, color: AMBER, isTextBox: true, margin: 0 });
s.addNotes("Introduce the team and the one-line idea: a team of specialised AI agents that turns a plain-language requirement into a tested, reviewed code change on a safe branch, handed to a human for approval.");

// ======================= Slide 1: Objectives / Problem =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Problem Statement and Objectives", "Most real software work changes an existing codebase without breaking it");
card(s, 0.5, 1.35, 4.35, 3.65, "The problem",
  "A request like \"Add password reset without affecting existing features\" needs many linked steps: read the repository, find the right files and tests, edit code, run tests, and recover from failures.\n\nSingle LLM calls produce plausible text, not a verified change - and they quietly break existing behaviour.",
  CORAL, { bodySize: 13, badge: "!" });
card(s, 5.15, 1.35, 4.35, 3.65, "Objectives",
  null, TEAL, { badge: "✓" });
s.addText(bullets([
  "Convert a natural-language requirement into a validated code change",
  "Ground every change in the real repository (RAG)",
  "Prove correctness with tests and detect regressions",
  "Recover from failures by diagnosis and re-planning",
  "Keep a human approval gate before any commit",
  "Compare architectures on reliability, latency and cost",
], 12.5), { x: 5.3, y: 1.9, w: 4.05, h: 3.0, fontFace: BODY, color: INK, valign: "top", isTextBox: true, margin: 0, paraSpaceAfter: 4 });
s.addNotes("Problem: changing existing code safely. Objective: a multi-agent system that produces a change plus evidence (tests, diff, review), and a study of how architecture choice affects reliability, latency and cost.");

// ======================= Slide 2: Literature Review =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Literature Review and Existing Systems");
const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 11 } });
const litRows = [
  [hdr("Work / Product"), hdr("Approach"), hdr("Limitation (gap)")],
  ["SWE-bench (Jimenez et al., 2024)", "Benchmark of real GitHub issues", "G1: evaluates models, does not build a validated workflow"],
  ["SWE-agent (Yang et al., 2024)", "Single agent + agent-computer interface", "G2: one agent; no explicit diagnosis / re-planning roles"],
  ["MetaGPT (Hong et al., 2024)", "Multi-agent roles via SOPs", "G3: mostly greenfield generation, weak repo grounding"],
  ["ReAct (Yao et al., 2023)", "Interleaved reasoning and acting", "G2: single-agent loop, no shared state"],
  ["RAG (Lewis et al., 2020)", "Retrieval-augmented generation", "G3: text retrieval, not code-structure aware"],
  ["Blackboard model (Nii, 1986)", "Shared-state problem solving", "G4: pre-LLM; not evaluated for LLM agents"],
  ["AI coding assistants (e.g. Copilot)", "IDE suggestions / chat agents", "G5: limited per-step evidence, cost and latency transparency"],
];
s.addTable(litRows, { x: 0.5, y: 1.05, w: 9.0, colW: [2.6, 2.75, 3.65], fontFace: BODY, fontSize: 10.5, color: INK,
  border: { type: "solid", color: "D5DEE4", pt: 0.75 }, fill: { color: WHITE }, rowH: 0.36, valign: "middle" });
box(s, 0.5, 4.45, 9.0, 0.6, "Gap: no open system combines repo-grounded multi-agent roles, test-verified changes with regression checks, failure-driven re-planning, human approval, and measured per-agent cost/latency.", LIGHT, INK, 11.5, false);
s.addNotes("Each row names a gap (G1-G5). Our requirements on the next slide map back to these gaps and to the problem statement.");

// ======================= Slide 3: Agent Requirements =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Agent Requirements and Traceability", "Every agent exists because of a requirement; every requirement traces to a gap (G) or the problem statement (PS)");
const reqRows = [
  [hdr("Req."), hdr("Requirement"), hdr("Source"), hdr("Agent / component")],
  ["R1", "Turn plain-language request into tasks + acceptance criteria", "PS", "Planner (CrewAI)"],
  ["R2", "Ground changes in the actual repository", "G3", "Repository/RAG agent (AST chunks)"],
  ["R3", "Edit code only in a sandbox / agentic branch", "PS", "Coder (ADK) + MCP repo tool"],
  ["R4", "Prove correctness, detect regressions", "G1, G5", "Test agent (ADK) + MCP test runner"],
  ["R5", "Diagnose failures and decide: more context / fix / stop", "G2", "Debugger (ADK) + Manager (CrewAI)"],
  ["R6", "Shared, consistent view of the run", "G4", "Blackboard + A2A envelopes"],
  ["R7", "Human approval before any commit", "PS", "HITL gate"],
  ["R8", "Measure per-agent latency, tokens, cost", "G5", "Callbacks + usage ledger"],
];
s.addTable(reqRows, { x: 0.5, y: 1.3, w: 9.0, colW: [0.6, 4.2, 0.9, 3.3], fontFace: BODY, fontSize: 10.5, color: INK,
  border: { type: "solid", color: "D5DEE4", pt: 0.75 }, rowH: 0.38, valign: "middle" });
s.addNotes("R1-R8 map to gaps G1-G5 from the literature slide or to the problem statement. R5 is the core research contribution: failure-driven re-planning.");

// ======================= Slide 4: MAS Architecture =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Multi-Agent Architecture (Hierarchical)", "Chosen: hierarchical - Manager on top, specialists below, shared Blackboard");
box(s, 0.5, 1.35, 1.35, 0.7, "Developer requirement", SLATE, WHITE, 10);
box(s, 2.1, 1.35, 1.35, 0.7, "Guardrails (injection, toxicity)", CORAL, WHITE, 10);
box(s, 3.7, 1.25, 2.6, 0.9, "Manager + Planner (CrewAI)\nplan, acceptance criteria, decisions", NAVY, WHITE, 10.5);
arrow(s, 1.85, 1.7, 2.1, 1.7); arrow(s, 3.45, 1.7, 3.7, 1.7);
const workers = [["RAG agent\n(ADK)", 0.5], ["Coder\n(ADK)", 2.3], ["Test agent\n(ADK)", 4.1], ["Debugger\n(ADK)", 5.9]];
for (const [t, x] of workers) { box(s, x, 2.75, 1.55, 0.7, t, TEAL, WHITE, 10.5); arrow(s, 5.0, 2.15, x + 0.78, 2.75, MUTED); }
box(s, 7.75, 2.75, 1.75, 0.7, "Auditor (CrewAI)\ndelivery package", NAVY, WHITE, 10);
box(s, 0.5, 3.75, 7.0, 0.55, "Shared Blackboard: requirement | plan | context | patches | test reports | diagnoses | decisions | status", LIGHT, INK, 10.5, true);
box(s, 0.5, 4.45, 3.4, 0.6, "MCP tools: repo/sandbox/Git + pytest runner", AMBER, INK, 10.5);
box(s, 4.1, 4.45, 3.4, 0.6, "Human approval gate -> commit on agentic/* branch", SLATE, WHITE, 10.5);
box(s, 7.75, 3.75, 1.75, 1.3, "Re-plan loop\nTests fail -> diagnose -> Manager decides -> act -> re-test", CORAL, WHITE, 9.5);
s.addNotes("Orchestration: the CrewAI Manager delegates to ADK workers through A2A envelopes; all agents read and write the Blackboard. When tests fail, the Debugger diagnoses, the Manager decides (more context / code fix / test fix / stop), the Planner re-plans, then tests re-run. Retry limit 3.");

// ======================= Slide 5: Architecture Evaluation =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Architecture Evaluation", "Same agents, tools, model and tasks - only the coordination pattern changes");
if (haveResults) {
  const rows = [[hdr("Configuration"), hdr("Success"), hdr("Time (s)"), hdr("Tokens"), hdr("Cost ($)"), hdr("Coverage"), hdr("Judge /5"), hdr("Retr. P@5")]];
  for (const r of archRows) rows.push([ARCH_NAMES[r.a], `${r.ok}/${r.runs}`, fmt(r.time), String(r.tokens), fmt(r.cost, 5),
    r.cov == null ? "-" : `${fmt(r.cov)}%`, fmt(r.judge, 2), fmt(r.prec, 2)]);
  s.addTable(rows, { x: 0.5, y: 1.3, w: 9.0, colW: [2.75, 0.8, 0.85, 0.85, 1.0, 0.95, 0.9, 0.9], fontFace: BODY, fontSize: 10, color: INK,
    border: { type: "solid", color: "D5DEE4", pt: 0.75 }, rowH: 0.34, valign: "middle" });
} else {
  box(s, 0.5, 1.35, 9.0, 1.4, "Live comparison results are added here after the evaluation runs (scripts/run_live_evaluation.py).", LIGHT, INK, 13, false);
}
card(s, 0.5, 3.25, 2.1, 1.8, "C1 chosen", "Manager re-plans from a diagnosis; one consistent shared state; high coverage.", NAVY, { bodySize: 10, headSize: 12 });
card(s, 2.8, 3.25, 2.1, 1.8, "C2", "Cheapest and fastest, but no repair loop and the lowest test coverage.", MUTED, { bodySize: 10, headSize: 12 });
card(s, 5.1, 3.25, 2.1, 1.8, "C3 P2P", "Same quality as C1 here; no global view for re-planning decisions.", TEAL, { bodySize: 10, headSize: 12 });
card(s, 7.4, 3.25, 2.1, 1.8, "C4 local LLM", "Free local coder, better retrieval; slower, needed repair cycles.", AMBER, { bodySize: 10, headSize: 12 });
s.addNotes("Why hierarchical: failures are the norm in real repos; only C1 classifies the failure and can choose a different remedy (fetch context vs fix vs stop). All 40 planned runs were completed live on gemini-3.5-flash. C1, C3 and C4 succeeded 10/10; the sequential C2 failed 2/10 (T7, T8) because it has no repair loop, although it is the cheapest and fastest. C3 matched C1 here; C1 keeps one global view for re-planning. C4 (local Ollama coder) needed repair cycles more often but always recovered, and its learned embeddings retrieved better context.");

// ======================= Slide 6: MCP =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Agent-Tool Communication via MCP", "Two FastMCP servers expose structured, sandbox-scoped tools");
const agentsL = ["RAG agent", "Coder", "Test agent", "Debugger"];
agentsL.forEach((t, i) => box(s, 0.5, 1.4 + i * 0.85, 1.8, 0.6, t, TEAL, WHITE, 11));
box(s, 3.0, 2.1, 1.9, 1.6, "MCP client\n(JSON-RPC tool calls,\ntyped results)", NAVY, WHITE, 11);
agentsL.forEach((t, i) => arrow(s, 2.3, 1.7 + i * 0.85, 3.0, 2.9));
card(s, 5.6, 1.3, 3.9, 1.85, "mcp-repository-tool", "create sandbox | clone repo | read / list files | apply patch | git branch, status, diff | RAG index + query | gated commit | PR publish", AMBER, { bodySize: 10.5 });
card(s, 5.6, 3.3, 3.9, 1.75, "mcp-test-runner", "Python syntax check | sandboxed pytest run with timeout | per-test report | line coverage (coverage.py)", AMBER, { bodySize: 10.5 });
arrow(s, 4.9, 2.6, 5.6, 2.2); arrow(s, 4.9, 3.2, 5.6, 4.1);
s.addNotes("Agents never touch the filesystem directly. Every file operation goes through the repository MCP server, which resolves paths inside the run's sandbox and blocks traversal. Test results come back as typed Pydantic reports.");

// ======================= Slide 7: Messaging =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Messaging Pattern and Agent-to-Agent Communication", "Blackboard (shared state) + A2A envelopes across CrewAI and ADK");
card(s, 0.5, 1.35, 4.35, 1.9, "Blackboard architecture (C1)", "Versioned shared state (v0 -> vN) with an RLock and observer callbacks. Agents read the plan, context, patches, test reports, diagnoses and decisions from one place.", NAVY, { bodySize: 11 });
card(s, 0.5, 3.4, 4.35, 1.65, "A2A envelopes", "Typed message with sender/recipient framework, correlation id and SHA-256 signature. TASK_DELEGATION -> TASK_RESULT; REPLAN_REQUEST.", TEAL, { bodySize: 11 });
s.addText("Cross-framework A2A participants", { x: 5.15, y: 1.35, w: 4.35, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: INK, isTextBox: true, margin: 0 });
const a2a = [["CrewAI", "Manager, Planner, Auditor"], ["Google ADK", "Coder, Test agent, Debugger, RAG agent"]];
a2a.forEach(([f, t], i) => { box(s, 5.15, 1.8 + i * 0.85, 1.4, 0.65, f, i ? TEAL : NAVY, WHITE, 11); s.addText(t, { x: 6.7, y: 1.8 + i * 0.85, w: 2.8, h: 0.65, fontFace: BODY, fontSize: 11.5, color: INK, valign: "middle", isTextBox: true, margin: 0 }); });
box(s, 5.15, 3.6, 4.35, 1.45, "7 agents on the A2A bus (3 CrewAI + 4 ADK).\nCompared: Blackboard (C1) vs peer-to-peer (C3): each agent receives all state in the message and sees nothing else; no Manager hub.", LIGHT, INK, 10.5, false);
s.addNotes("Be precise in the viva: A2A here is a typed, signed envelope protocol over an in-process bus; agents are not separate network services. C3 is a true peer-to-peer configuration: every hand-off carries the full state, each agent rebuilds its view only from the message it received, and the Debugger routes failures itself.");

// ======================= Slide 8: Security =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Securing the Multi-Agent System");
card(s, 0.5, 1.05, 2.9, 3.95, "Guardrails", null, CORAL, {});
s.addText(bullets(["Prompt-injection patterns", "Toxic / abusive language check", "AST scan of generated code (os.system, eval, subprocess)", "Path guard: no traversal outside sandbox", "Push only to agentic/* branches"], 11.5),
  { x: 0.65, y: 1.65, w: 2.65, h: 3.3, fontFace: BODY, color: INK, valign: "top", isTextBox: true, margin: 0, paraSpaceAfter: 4 });
card(s, 3.55, 1.05, 2.9, 3.95, "Safeguards", null, AMBER, {});
s.addText(bullets(["Isolated sandbox per run", "Human approval gate before commit", "Retry limit + no-progress stop rule", "Quota errors fail the run (no fake success)", "Secrets only via environment / cloud secrets"], 11.5),
  { x: 3.7, y: 1.65, w: 2.65, h: 3.3, fontFace: BODY, color: INK, valign: "top", isTextBox: true, margin: 0, paraSpaceAfter: 4 });
card(s, 6.6, 1.05, 2.9, 3.95, "Callbacks & audit", null, TEAL, {});
s.addText(bullets(["on_agent_start / finish / error", "on_tool_start / end", "Tool-output schema validation (every MCP call)", "Per-agent usage ledger (tokens, latency)", "SHA-256 chained provenance log"], 11.5),
  { x: 6.75, y: 1.65, w: 2.65, h: 3.3, fontFace: BODY, color: INK, valign: "top", isTextBox: true, margin: 0, paraSpaceAfter: 4 });
s.addNotes("Defense in depth: input guardrails before any LLM call, output guardrails (AST scan) before delivery, sandbox and branch protection for actions, a human gate before commit, and callbacks/provenance for audit.");

// ======================= Slide 9: Experimental Design =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Experimental Design for Validation", "Live runs with real Gemini calls; every run stored with per-agent telemetry");
const statN = haveResults ? String(totalRuns) : "0";
[[statN, "live runs"], ["4+4", "configs + prompt/persona variants"], [String(nTasks || 0), "change tasks"], ["129", "automated tests"]].forEach(([n, l], i) => {
  const x = 0.5 + i * 2.3;
  s.addText(n, { x, y: 1.3, w: 2.1, h: 0.8, fontFace: HEAD, fontSize: 40, bold: true, color: TEAL, align: "left", isTextBox: true, margin: 0 });
  s.addText(l, { x, y: 2.1, w: 2.1, h: 0.35, fontFace: BODY, fontSize: 12, color: MUTED, isTextBox: true, margin: 0 });
});
card(s, 0.5, 2.75, 4.35, 2.3, "Variance and coverage", "Content variance: different features and edge cases per task. Architecture variance: every task under C1-C4. Model variance: Gemini vs local Ollama coder (C4). Offline: 129 tests incl. failure injection (missing import, logic bug, regression, syntax error, quota).", NAVY, { bodySize: 10.5 });
card(s, 5.15, 2.75, 4.35, 2.3, "Prompt & persona variance (C1, T1-T2)", "All 8 variant runs passed. ReAct (tool-using): most tokens (~10.2k vs 6.5k) and the only repair cycle. Few-shot: same cost as standard. Strict Architect: most tests written, slowest. Permissive Prototyper: cheapest. Judge 4.6-5.0 for all.", TEAL, { bodySize: 10.5 });
s.addNotes("Main matrix: 10 tasks x 4 configurations = 40 live runs, all completed. Plus 8 runs varying only the Coder prompt (standard / ReAct / few-shot) or the team persona (Balanced / Strict / Permissive) on C1. The free tier allows 20 requests per model per project per day, so runs were spread across API projects; quota-interrupted runs were re-run and never counted. Two tasks per variant gives indicative, not significant, differences.");

// ======================= Slide 10: Validation =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Validating the MAS", "Is it working, how fast is each agent, and how is output judged?");
if (agentNames.length) {
  s.addChart(pres.charts.BAR, [{ name: "Avg latency (s)", labels: agentNames, values: agentAvg }], {
    x: 0.5, y: 1.3, w: 5.0, h: 3.75, barDir: "bar", showTitle: true, title: "Per-agent response time, C1 (s)", titleFontSize: 12, titleFontFace: "Calibri",
    chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelFontFace: "Calibri",
    catAxisLabelFontSize: 9, valAxisLabelFontSize: 9, catAxisLabelFontFace: "Calibri", valAxisLabelFontFace: "Calibri",
    catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, valGridLine: { color: "E3E8EC", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  });
} else {
  box(s, 0.5, 1.3, 5.0, 3.75, "Per-agent response-time chart is generated from the live runs.", LIGHT, INK, 12, false);
}
const c1 = archRows.find(r => r.a === "C1");
card(s, 5.75, 1.3, 3.75, 1.75, "Working as expected?", c1 ? `C1 succeeded in ${c1.ok}/${c1.runs} live runs; avg end-to-end ${c1.time.toFixed(1)} s, ${c1.tokens} tokens per run.` : "Success rate per architecture is filled from live runs.", TEAL, { bodySize: 11 });
card(s, 5.75, 3.2, 3.75, 1.85, "How output is judged", "Tests decide pass/fail (existing + generated suites, regression check, coverage). LLM-as-Judge (ADK, different model) scores plan, code and tests 1-5: avg " + (archRows.find(r => r.a === "C1") ? fmt(archRows.find(r => r.a === "C1").judge, 2) : "-") + " for C1.", NAVY, { bodySize: 11 });
s.addNotes("Response time is measured per agent by the usage ledger, from real provider calls. Correctness is judged by executable tests; the LLM-as-Judge agent (gemini-3.5-flash-lite, a different model from the generator) rates plan quality, code correctness, code quality and test adequacy against a fixed rubric.");

// ======================= Slide 11: Team =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Team Responsibility and Contribution");
card(s, 0.5, 1.05, 4.35, 3.35, "Muskan Patni (230534)", null, TEAL, {});
s.addText(bullets(["Repository/RAG agent (AST chunking, retrieval)", "Coding agent and Testing/Validation agent", "MCP tool layer, sandbox and Git integration", "Experiment runner and LLM evaluation", "Deployment"], 12),
  { x: 0.65, y: 1.65, w: 4.05, h: 2.7, fontFace: BODY, color: INK, valign: "top", isTextBox: true, margin: 0, paraSpaceAfter: 4 });
card(s, 5.15, 1.05, 4.35, 3.35, "Vaishnavi Singh (230508)", null, NAVY, {});
s.addText(bullets(["Manager/Planner agent and orchestration", "Re-planning logic and Debugging/Review agent", "Shared Blackboard and A2A communication", "Observability, logging, tracing, HITL gate", "System-level evaluation"], 12),
  { x: 5.3, y: 1.65, w: 4.05, h: 2.7, fontFace: BODY, color: INK, valign: "top", isTextBox: true, margin: 0, paraSpaceAfter: 4 });
box(s, 0.5, 4.55, 9.0, 0.5, "Both: literature review, task selection, integration, pilot testing, evaluation, documentation and presentation", LIGHT, INK, 11.5, false);
s.addNotes("Matches the division of work in the Phase 1 synopsis, Section 2.2.");

// ======================= Slide 12: Screenshots =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Demonstration Screenshots", "Deployed DevPilot AI dashboard");
const shots = ["Knowledge Base tab (repository + RAG)", "Ask tab: run in progress", "Diff, tests and approval gate", "Evaluation tab: runs and metrics"];
shots.forEach((t, i) => {
  const x = 0.5 + (i % 2) * 4.6, y = 1.3 + Math.floor(i / 2) * 1.95;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.4, h: 1.8, rectRadius: 0.06, fill: { color: LIGHT }, line: { color: "C9D3DA", dashType: "dash", width: 1 } });
  s.addText(`[Replace with screenshot]\n${t}`, { x, y, w: 4.4, h: 1.8, fontFace: BODY, fontSize: 12, color: MUTED, align: "center", valign: "middle", isTextBox: true, margin: 4 });
});
s.addNotes("Replace each box with a screenshot taken from the DEPLOYED app (not localhost): Insert > Picture, then delete the placeholder box.");

// ======================= Slide 13: Conclusions =======================
s = pres.addSlide({ masterName: "DARK" });
s.addText("Conclusions", { x: 0.6, y: 0.4, w: 8.8, h: 0.6, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, isTextBox: true, margin: 0 });
const concl = [
  ["Working MAS", "CrewAI + Google ADK agents coordinate over a Blackboard and A2A to deliver test-verified changes."],
  ["Self-correcting", "Failures are diagnosed and the Manager re-plans: more context, code fix, test fix or stop."],
  ["Safe by design", "Guardrails, sandbox, branch protection and a human gate before every commit."],
  ["Measured honestly", "Per-agent latency, tokens and cost from real calls; quota failures are never counted as success."],
];
concl.forEach(([h, b], i) => {
  const x = 0.6 + (i % 2) * 4.45, y = 1.3 + Math.floor(i / 2) * 1.75;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.2, h: 1.55, rectRadius: 0.08, fill: { color: "17384F" }, line: { color: "17384F" } });
  s.addText(h, { x: x + 0.2, y: y + 0.15, w: 3.8, h: 0.35, fontFace: BODY, fontSize: 14, bold: true, color: AMBER, isTextBox: true, margin: 0 });
  s.addText(b, { x: x + 0.2, y: y + 0.55, w: 3.8, h: 0.9, fontFace: BODY, fontSize: 12, color: WHITE, valign: "top", isTextBox: true, margin: 0 });
});
s.addText("Future work: more runs per configuration for significance, larger local models for C4, multi-language repositories, A2A agents as separate network services.", { x: 0.6, y: 4.9, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 11, italic: true, color: "CADCE6", isTextBox: true, margin: 0 });
s.addNotes("Summarise contributions, then state limitations honestly as future work.");

// ======================= Slide 14: References =======================
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "References");
const refs = [
  "[1] C. E. Jimenez et al., \"SWE-bench: Can Language Models Resolve Real-World GitHub Issues?\", ICLR 2024.",
  "[2] J. Yang et al., \"SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering\", NeurIPS 2024.",
  "[3] S. Hong et al., \"MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework\", ICLR 2024.",
  "[4] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed., Pearson, 2020.",
  "[5] S. Yao et al., \"ReAct: Synergizing Reasoning and Acting in Language Models\", ICLR 2023.",
  "[6] J. Wei et al., \"Chain-of-Thought Prompting Elicits Reasoning in LLMs\", NeurIPS 2022.",
  "[7] P. Lewis et al., \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks\", NeurIPS 2020.",
  "[8] Model Context Protocol, documentation and specification.",
  "[9] H. P. Nii, \"The Blackboard Model of Problem Solving\", AI Magazine 7(2), 1986.",
  "[10] L. Zheng et al., \"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena\", NeurIPS 2023.",
  "[11] CrewAI documentation; Google Agent Development Kit (ADK) documentation; Agent2Agent (A2A) protocol.",
];
s.addText(refs.map((t, i) => ({ text: t, options: { breakLine: i < refs.length - 1 } })),
  { x: 0.5, y: 1.0, w: 9.0, h: 4.1, fontFace: BODY, fontSize: 10.5, color: INK, valign: "top", isTextBox: true, margin: 0, paraSpaceAfter: 3 });

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote " + f));
