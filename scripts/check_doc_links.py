"""Check that backtick-quoted repo paths referenced in docs exist.

Pragmatic rules (docs also mention runtime outputs, API routes, filename
lists and keyword lists — those are skipped):
- skip API routes (start with "/")
- skip runtime artifacts (results/, dist/, build/, ml/results, theta_*)
- handle extension alternatives (`a.docx/.pdf` -> check `a.docx`)
- skip filename lists where every segment has an extension
- skip keyword lists (>=3 segments, no extension anywhere)
- skip short alternatives like `m2_tilt_x/y` (last segment has no dot, len<=2)

Usage: python3 scripts/check_doc_links.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Windows consoles default to GBK and raise UnicodeEncodeError on the Chinese
# doc paths printed below; force UTF-8 so the checker is usable everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):      # pragma: no cover - py<3.7/redirected
        pass

SKIP_DIRS = {"docs/archive", "archive", "results", ".git", "build", "dist",
             "node_modules", ".venv", ".pytest_cache"}
# roadmap doc lists planned (not yet existing) files by design
SKIP_FILES = {"docs/后续优化方向.md"}
SKIP_PREFIXES = ("--", "-", "python", "pip", "uvicorn", "ctest", "cmake", "git ",
                 "QT_", "http://", "https://", "~/", "$", "/")
RUNTIME_FIRST = {"results", "dist", "build", "ml", "theta_XXXdeg"}
SKIP_EXACT = {"web/api/main.py", "web/ui/index.html", ".NET/pythonnet",
              "t/states", "pybind11/eigen.h", "ParameterSpace/ControlSpace",
              "parse_zmx_bytes/file", "case_detail/environment_sensitivity",
              "data/ccd_pipeline/", "analyze/control", "InverseSolver/Solution", "baseline/", "web/serve.py", "web/index.html",
              "raw/", "calib/"}

BACKTICK = re.compile(r"`([^`\n]{3,120})`")
EXT_ALT = re.compile(r"^(.*?\.[A-Za-z0-9]+)/(?:\.?[A-Za-z0-9]+)+$")


def looks_like_path(tok: str) -> bool:
    if tok.startswith(SKIP_PREFIXES) or " " in tok.strip():
        return False
    if re.match(r"^[A-Za-z]:[\\/]", tok):
        return False
    if not ("/" in tok or "\\" in tok):
        return False
    if any(c in tok for c in "*{}[]()<>=!|"):
        return False
    if re.match(r"^\d", tok):
        return False
    return True


def check_one(tok: str):
    """Return (should_check, candidate) — candidate None means skip."""
    cand = tok.replace("\\", "/").rstrip("/")
    first = cand.split("/", 1)[0]
    if first in RUNTIME_FIRST:
        return False, None
    m = EXT_ALT.match(cand)
    if m:
        cand = m.group(1)
    segs = cand.split("/")
    if all("." in s for s in segs):
        return False, None  # filename list
    if "." not in cand and len(segs) >= 3:
        return False, None  # keyword list
    if "." not in segs[-1] and len(segs[-1]) <= 2:
        return False, None  # x/y alternatives
    return True, cand


def main() -> int:
    problems = []
    for md in sorted(ROOT.rglob("*.md")):
        rel = md.relative_to(ROOT).as_posix()
        if rel in SKIP_FILES or any(rel.startswith(d + "/") or rel == d for d in SKIP_DIRS):
            continue
        head = md.read_text(encoding="utf-8")[:200]
        if rel.startswith("docs/tutorials/") and "已归档" in head:
            continue
        for tok in BACKTICK.findall(md.read_text(encoding="utf-8")):
            tok = tok.strip()
            if tok in SKIP_EXACT or not looks_like_path(tok):
                continue
            should, cand = check_one(tok)
            if not should:
                continue
            if (ROOT / cand).exists() or (md.parent / cand).exists():
                continue
            problems.append(f"{rel}: `{tok}`")
    if problems:
        print("MISSING DOC REFERENCES:")
        for p in problems:
            print(" ", p)
        return 1
    print("doc links OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
