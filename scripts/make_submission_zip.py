"""Builds the clean source-code ZIP for submission.

Excludes secrets (.env), virtual environments, caches, per-run sandboxes, the regenerated
vector index and Git metadata. Includes .env.example so graders can configure their own key.

Usage:  .venv\\Scripts\\python.exe scripts\\make_submission_zip.py [output.zip]
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "DevPilot_AI_Source_Code.zip"

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules",
}
EXCLUDED_PREFIXES = (
    ".venv_",
    "sandbox_workspace/runs",
    "sandbox_workspace/chroma_db",
    "sandbox_workspace/temp",
)
EXCLUDED_FILES = {".env"}
EXCLUDED_SUFFIXES = {".pyc", ".log"}


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    parts = rel.split("/")
    if any(p in EXCLUDED_DIRS for p in parts):
        return False
    if any(rel.startswith(pref) or parts[0].startswith(pref) for pref in EXCLUDED_PREFIXES):
        return False
    if path.name in EXCLUDED_FILES or path.suffix in EXCLUDED_SUFFIXES:
        return False
    return True


def main() -> None:
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and included(p))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, Path("MultiAgent-Software-Engineering") / f.relative_to(ROOT))
    names = zipfile.ZipFile(OUT).namelist()
    assert not any(n.endswith("/.env") for n in names), ".env leaked into the ZIP"
    print(f"Wrote {OUT} ({len(names)} files, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
