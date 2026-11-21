"""Filesystem locations shared by the Qt GUI.

Split from desktop/gui.py (old lines 22-24 and 105-126): repo root plus the
writable-directory probing used by every export path.  ``desktop.gui`` keeps
re-exporting ``_writable_dir`` / ``_default_export_dir`` because
desktop/tests/test_gui_helpers.py imports them from the facade.
"""
from __future__ import annotations

from pathlib import Path

# desktop/ui/paths.py -> parents[2] == repo root (gui.py sat one level higher
# and used parents[1]; the resolved directory is identical).
ROOT = Path(__file__).resolve().parents[2]


def _writable_dir(*candidates) -> Path:
    """First candidate directory that can actually be written to."""
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return p
        except Exception:
            continue
    return Path.cwd()


def _default_export_dir() -> Path:
    docs = Path.home() / "Documents"
    return _writable_dir(docs / "TelescopeEngineering" / "exports",
                         ROOT / "results" / "exports",
                         Path.cwd() / "telescope_exports")
