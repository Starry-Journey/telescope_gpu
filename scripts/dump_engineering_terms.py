r"""Dump the 6-term observability / astigmatism plan for the data package."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from products.sensitivity.service import engineering_terms_sensitivity  # noqa: E402

d = engineering_terms_sensitivity(source="abaqus")
out = ROOT / "data" / "fea_cases" / "abaqus_m1" / "engineering_terms_sensitivity.json"
out.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
print("targets:", d["targets"])
print("rank/cond:", d["rank"], round(d["condition"], 2))
print("dropped rows/cols:", d["dropped_rows"], d["dropped_cols"])
plan = d["astigmatism_plan"]
print("astig:", round(plan["magnitude_nm"], 4), "nm",
      "axis", round(plan["astigmatism_axis_deg"], 2), "deg",
      "adjusters", plan["adjuster_count"])
