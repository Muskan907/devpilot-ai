"""Unit and integration tests for DevPilot AI SaaS Dashboard and Evaluation Components."""

from pathlib import Path
import py_compile
import pytest

from evaluation.comparison import architecture_comparator
from evaluation.experiment_db import ExperimentStorage
from orchestration import AgenticPatternType, PersonaType
from evaluation import BENCHMARK_TASKS


def test_dashboard_script_compilation():
    app_path = Path("dashboard/app.py")
    assert app_path.exists(), "dashboard/app.py must exist"
    # Verify file compiles cleanly without syntax errors
    compiled = py_compile.compile(str(app_path), doraise=True)
    assert compiled is not None


def test_dashboard_css_exists_and_contains_brand_tokens():
    css_path = Path("dashboard/styles.css")
    assert css_path.exists(), "dashboard/styles.css must exist"
    content = css_path.read_text(encoding="utf-8")

    # Brand color tokens & typography
    assert "#0B0E14" in content, "Primary dark background (#0B0E14) missing"
    assert "#2F81F7" in content, "Electric blue accent (#2F81F7) missing"
    assert "#30363D" in content, "Subtle border color (#30363D) missing"
    assert "Inter" in content, "Inter typography missing"
    assert "JetBrains Mono" in content, "JetBrains Mono monospace font missing"

    # Component styling
    assert ".saas-card" in content, ".saas-card class missing"
    assert ".saas-metric-box" in content, ".saas-metric-box class missing"
    assert ".badge-blue" in content, "Badge classes missing"
    assert ".diff-container" in content, "Diff container styling missing"
    assert ".timeline-track" in content, "Timeline track styling missing"
    assert ".context-banner" in content, "Context banner styling missing"


def test_c1_c4_conditions_definitions_completeness():
    conditions = architecture_comparator.get_c1_c4_conditions_definitions()
    assert len(conditions) == 4, f"Expected 4 experimental conditions, got {len(conditions)}"

    cond_strings = [c["condition"] for c in conditions]
    assert any("C1" in s for s in cond_strings), "Condition C1 (Proposed System) missing"
    assert any("C2" in s for s in cond_strings), "Condition C2 (Monolithic Baseline) missing"
    assert any("C3" in s for s in cond_strings), "Condition C3 (Protocol Ablation) missing"
    assert any("C4" in s for s in cond_strings), "Condition C4 (Multi-Model Evaluation) missing"

    # Verify communication paradigms
    c1 = next(c for c in conditions if "C1" in c["condition"])
    assert "Blackboard" in c1["communication"]
    assert "A2A" in c1["communication"]

    c3 = next(c for c in conditions if "C3" in c["condition"])
    assert "Peer-to-Peer" in c3["communication"] or "P2P" in c3["communication"]


def test_c4_model_family_comparison_matrix():
    matrix = architecture_comparator.get_c4_model_comparison_matrix()
    assert len(matrix) >= 4, "C4 matrix must contain at least 4 models"

    models = [m["Model"] for m in matrix]
    assert "gemini-3.8-flash" in models or any("3.8-flash" in m for m in models)
    assert "gemini-1.5-pro" in models or any("1.5-pro" in m for m in models)
    assert "gemini-2.0-flash" in models or any("2.0-flash" in m for m in models)
    assert "gpt-4o" in models or any("gpt-4o" in m for m in models)

    # Check pricing and context columns
    for row in matrix:
        assert "Prompt Rate ($/1M)" in row
        assert "Completion Rate ($/1M)" in row
        assert "Context Window" in row
        assert "Provider" in row


def test_benchmark_matrix_dimensions():
    # Legacy persona matrix: 5 tasks x patterns x 3 personas; synopsis campaign: C1-C4 configurations.
    tasks = BENCHMARK_TASKS
    architectures = list(AgenticPatternType)
    personas = list(PersonaType)

    assert len(tasks) == 5, f"Expected 5 benchmark tasks, got {len(tasks)}"
    assert len(personas) == 3, f"Expected 3 developer personas, got {len(personas)}"
    synopsis_configs = {
        AgenticPatternType.HIERARCHICAL_BLACKBOARD_A2A,  # C1
        AgenticPatternType.SEQUENTIAL_BASELINE,          # C2
        AgenticPatternType.HIERARCHICAL_P2P_A2A,         # C3
        AgenticPatternType.HIERARCHICAL_CROSS_MODEL,     # C4
    }
    assert synopsis_configs.issubset(set(architectures))
    assert AgenticPatternType.CRITIC_REFLECTION_LOOP in architectures  # extra pattern

    total_cells = len(tasks) * len(architectures) * len(personas)
    assert total_cells == 5 * len(architectures) * 3


def test_dashboard_experiment_storage_querying(tmp_path):
    db_file = tmp_path / "dashboard_eval_test.db"
    storage = ExperimentStorage(db_path=str(db_file))

    # Test empty state
    runs = storage.get_all()
    recent = storage.get_recent_runs(limit=10)
    prs = storage.get_pull_requests()
    assert len(runs) == 0
    assert len(recent) == 0
    assert len(prs) == 0

    # Test summary stats empty
    summary = storage.get_summary_statistics()
    assert summary["total_runs"] == 0
    assert summary["success_rate_percentage"] == 0.0


def test_dashboard_headless_apptest_run():
    """Verify entire Streamlit dashboard UI tree compiles and renders headlessly without exceptions."""
    from streamlit.testing.v1 import AppTest
    app_file = Path(__file__).parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised unexpected exception: {at.exception}"
    assert len(at.tabs) >= 4, f"Expected at least 4 tabs, found {len(at.tabs)}"


def test_dashboard_apptest_tab_content_and_widgets():
    """Verify AppTest inspects widgets, buttons, and tab structures across all 4 tabs."""
    from streamlit.testing.v1 import AppTest
    app_file = Path(__file__).parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=30)
    at.run()
    assert not at.exception

    # Verify buttons
    button_keys = [b.key for b in at.button]
    assert "btn_connect_repo" in button_keys
    assert "btn_rag_search" in button_keys
    assert "btn_run_pipeline" in button_keys
    assert "btn_run_compare" in button_keys

    # Verify selectboxes (Architecture, Persona, Benchmark presets, Task filter, etc.)
    assert len(at.selectbox) >= 4


