"""M1 meniscus FEA following the Zijinshan design document (Abaqus 2025).

Geometry/material come from 多终端通用望远镜详细设计方案20240408-修订.pdf,
表 2 主镜 M1 设计参数列表 (p.14) and §"主次镜系统采用 RC 系统" (p.13):

    顶点曲率 R = 7331 mm, 非球面系数 k = -1.02347
    最大口径 2560 mm, 通光口径 2500 mm (中孔 700 mm), 边沿厚度 260 mm
    镜厚比 9:1, 材料 微晶玻璃 (标准级), CTE < 1e-7 /K
    面形要求 RMS <= 1/40 lambda @632.8 nm  (= 15.82 nm)
    工作温度 15..25 degC (p.~ thermal section) -> uniform +/-5 K case

The document does NOT specify the whiffletree layout, the lightweighting/rib
pattern, or the exact material constants, so those stay ASSUMED and are
reported as such:
  * solid meniscus (constant 260 mm thickness following the conic front),
  * 36 axial pads (6-12-12-6) + 24 lateral pads, each spread over a node patch,
  * E = 90 GPa, nu = 0.24, rho = 2.53e-9 t/mm^3 (document gives only CTE).

Output: data/fea_cases/abaqus_design/<case>_top.csv  (same schema as the rest)
        plus a summary JSON with citations/assumptions.

Run (Windows):
  D:\\SIMULIA\\Commands\\abq2025.bat cae noGUI=scripts/abaqus_m1_design.py
"""
from __future__ import print_function

import csv
import json
import math
import os

from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from odbAccess import openOdb

executeOnCaeStartup()
Mdb()

OUT = os.path.abspath(os.environ.get(
    "TEL_ABAQUS_OUT", os.path.join(os.getcwd(), "data", "fea_cases",
                                   "abaqus_design")))
SCRATCH = os.path.join(OUT, "odb")
for d in (OUT, SCRATCH):
    if not os.path.isdir(d):
        os.makedirs(d)
os.chdir(SCRATCH)

# ---- document parameters (表 2, p.14) ----
R_VERTEX = 7331.0
CONIC = -1.02347
R_CLEAR = 1250.0        # 通光口径 2500
R_OUTER = 1280.0        # 最大口径 2560
R_HOLE = 350.0          # 中孔 700
THICK = 260.0           # 边沿厚度
# ---- ASSUMED material constants (document only fixes CTE) ----
E = 90000.0
NU = 0.24
RHO = 2.53e-9
ALPHA = 1.0e-7          # document: < 1e-7 /K
SEED = float(os.environ.get("TEL_ABAQUS_SEED", 40.0))
K_BOTTOM = float(os.environ.get("TEL_ABAQUS_KB", 1.0e6))
K_SIDE = float(os.environ.get("TEL_ABAQUS_KS", 5.0e5))
N_BOTTOM = ((375.0, 6), (625.0, 12), (875.0, 12), (1125.0, 6))   # ASSUMED
N_SIDE = 24                                                      # ASSUMED
R_PAD = 70.0
R_PAD_SIDE = 45.0
SPEC_RMS_NM = 632.8 / 40.0
CASES = (
    ("gravity_0deg", "gravity", (9810.0, 0.0, 0.0)),
    ("gravity_90deg", "gravity", (0.0, 0.0, -9810.0)),
    ("thermal_plus5K", "thermal", 5.0),
    ("thermal_minus5K", "thermal", -5.0),
)


def sag(r):
    """Conic sag for R, k (mm); r measured from the vertex."""
    if r <= 0.0:
        return 0.0
    denom = R_VERTEX * (1.0 + math.sqrt(
        max(1.0 - (1.0 + CONIC) * (r / R_VERTEX) ** 2, 1e-12)))
    return r * r / denom


Y_FRONT = lambda r: THICK               # flat top = optical surface (z = 260)
Y_BACK = lambda r: 0.0                  # flat back (z = 0)


def build_part(model):
    """Annular solid with the documented OD/ID/thickness (extrude).

    CAE BaseSolidRevolve fails in this Abaqus install ("实体旋转特征创建失败")
    for any profile, so the 106.6 mm conic sag is NOT represented here: the
    model is a flat annular disk with the documented outer diameter (2560 mm),
    central hole (700 mm) and edge thickness (260 mm).  This is recorded in the
    summary as an explicit limitation.
    """
    sketch = model.ConstrainedSketch(name="profile", sheetSize=3000.0)
    sketch.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R_OUTER, 0.0))
    sketch.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R_HOLE, 0.0))
    part = model.Part(name="M1", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sketch, depth=THICK)
    mat = model.Material(name="Sitall")
    mat.Elastic(table=((E, NU),))
    mat.Density(table=((RHO,),))
    mat.Expansion(table=((ALPHA,),))
    model.HomogeneousSolidSection(name="M1sec", material="Sitall",
                                  thickness=None)
    part.SectionAssignment(region=(part.cells,), sectionName="M1sec")
    part.seedPart(size=SEED, deviationFactor=0.2, minSizeFactor=0.2)
    elem = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
    part.setElementType(regions=(part.cells,), elemTypes=(elem,))
    part.generateMesh()
    return part


def nodes_in_ball(inst, cx, cy, cz, radius):
    labels = []
    r2 = radius * radius
    for n in inst.nodes:
        c = n.coordinates
        if (c[0] - cx) ** 2 + (c[1] - cy) ** 2 + (c[2] - cz) ** 2 <= r2:
            labels.append(n.label)
    return labels


def nodes_on_surface(inst, radius, y_of_r, tol=1.5):
    labels = []
    for n in inst.nodes:
        c = n.coordinates
        r = math.sqrt(c[0] * c[0] + c[2] * c[2])
        if abs(c[1] - y_of_r(r)) <= tol:
            labels.append(n.label)
    return labels


def add_supports(model, asm, inst):
    ef = model.rootAssembly.engineeringFeatures
    n_pad = 0
    for ring_r, count in N_BOTTOM:
        for k in range(count):
            ang = 2.0 * math.pi * k / count
            name = "PAD_%02d" % n_pad
            n_pad += 1
            cx, cz = ring_r * math.cos(ang), ring_r * math.sin(ang)
            labels = nodes_in_ball(inst, cx, 0.0, cz, R_PAD)
            if not labels:
                raise RuntimeError("no nodes near pad %s" % name)
            asm.Set(name=name, nodes=inst.nodes.sequenceFromLabels(tuple(labels)))
            ef.SpringDashpotToGround(
                name=name + "_D2", region=asm.sets[name], orientation=None,
                dof=3, springBehavior=ON, springStiffness=K_BOTTOM / len(labels))
    n_side_pad = 0
    for k in range(N_SIDE):
        ang = 2.0 * math.pi * k / N_SIDE
        name = "SIDE_%02d" % k
        n_side_pad += 1
        labels = []
        for tol in (2.0, 20.0, 40.0):
            labels = [n.label for n in inst.nodes
                      if abs(math.sqrt(n.coordinates[0] ** 2 +
                                       n.coordinates[1] ** 2) - R_OUTER) <= R_PAD_SIDE
                      and abs(math.atan2(math.sin(math.atan2(
                          n.coordinates[1], n.coordinates[0]) - ang),
                          math.cos(math.atan2(n.coordinates[1],
                                              n.coordinates[0]) - ang))) <= R_PAD_SIDE / R_OUTER
                      and abs(n.coordinates[2] - THICK / 2.0) <= tol]
            if labels:
                break
        if not labels:
            raise RuntimeError("no nodes near side pad %s" % name)
        asm.Set(name=name, nodes=inst.nodes.sequenceFromLabels(tuple(labels)))
        for dof in (1, 2):
            ef.SpringDashpotToGround(
                name="%s_D%d" % (name, dof), region=asm.sets[name],
                orientation=None, dof=dof, springBehavior=ON,
                springStiffness=K_SIDE / len(labels))
    return n_pad, n_side_pad


def add_load(model, asm, inst, kind, spec):
    model.StaticStep(name="load", previous="Initial", timePeriod=1.0)
    if kind == "gravity":
        model.Gravity(name="g", createStepName="load", comp1=spec[0],
                      comp2=spec[1], comp3=spec[2])
    else:
        asm.Set(name="ALL_NODES", nodes=inst.nodes)
        model.Temperature(name="dT", createStepName="load",
                          region=asm.sets["ALL_NODES"],
                          distributionType=UNIFORM,
                          crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                          magnitudes=(spec,))
    model.fieldOutputRequests["F-Output-1"].setValues(variables=("U",))
    return "load"


def make_case(tag, kind, spec):
    name = "M1D_" + tag
    if name in mdb.models:
        del mdb.models[name]
    model = mdb.Model(name=name)
    part = build_part(model)
    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    inst = asm.Instance(name="M1-1", part=part, dependent=ON)
    nb, ns = add_supports(model, asm, inst)
    step = add_load(model, asm, inst, kind, spec)
    job = mdb.Job(name=name, model=name, type=ANALYSIS, numCpus=1,
                  explicitPrecision=SINGLE)
    print("solving %s (pads %d/%d)" % (name, nb, ns))
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    return name, step


def export_top(name, step, tag):
    odb_path = os.path.join(SCRATCH, name + ".odb")
    if not os.path.exists(odb_path):
        raise RuntimeError("missing ODB " + odb_path)
    odb = openOdb(odb_path, readOnly=True)
    inst = odb.rootAssembly.instances["M1-1"]
    coords = {n.label: n.coordinates for n in inst.nodes}
    frame = odb.steps[step].frames[-1]
    u = frame.fieldOutputs["U"]
    rows = []
    for v in u.values:
        c = coords.get(v.nodeLabel)
        if c is None:
            continue
        if abs(c[2] - THICK) <= 1.5:            # top (optical) surface
            rows.append((c[0], c[1], c[2], v.data[0], v.data[1], v.data[2]))
    out_csv = os.path.join(OUT, "M1_design_%s_top.csv" % tag)
    with open(out_csv, "w") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x_mm", "y_mm", "z_mm", "u1_mm", "u2_mm", "dz_nm"])
        for x, y, z, u1, u2, u3 in rows:
            wr.writerow(["%.6f" % x, "%.6f" % y, "%.6f" % z,
                         "%.9f" % u1, "%.9f" % u2, "%.6f" % (u3 * 1.0e6)])
    odb.close()
    print("  %d front nodes -> %s" % (len(rows), out_csv))
    return len(rows)


summary = {
    "source_document": ("多终端通用望远镜详细设计方案20240408-修订.pdf, "
                        "表 2 主镜 M1 设计参数列表 (p.14), 主次镜系统 (p.13)"),
    "documented": {"R_vertex_mm": R_VERTEX, "conic": CONIC,
                   "outer_diameter_mm": 2 * R_OUTER,
                   "clear_aperture_mm": 2 * R_CLEAR,
                   "central_hole_mm": 2 * R_HOLE,
                   "edge_thickness_mm": THICK,
                   "thickness_ratio": "9:1",
                   "material": "微晶玻璃 (sitall, 标准级)",
                   "cte_per_K": "1e-7 (upper bound from text)",
                   "spec_rms_nm": SPEC_RMS_NM,
                   "thermal_range_K": "15..25 degC -> +/-5 K"},
    "limitations": ["conic sag (%.1f mm at the edge) not modelled: CAE "
                    "BaseSolidRevolve fails in this install, so the model is a "
                    "flat annulus with the documented OD/ID/thickness" % sag(R_OUTER)],
    "assumed": {"support_layout": "36 axial (6-12-12-6) + 24 lateral (NOT in text)",
                "lightweighting": "solid meniscus (no rib pattern in text)",
                "E_MPa": E, "nu": NU, "rho_t_mm3": RHO,
                "alpha_used_per_K": ALPHA,
                "pad_patch_mm": [R_PAD, R_PAD_SIDE]},
    "cases": {},
    "label": "FEA-PROVISIONAL",
}
print("OUT =", OUT, "seed =", SEED)
for tag, kind, spec in CASES:
    nm, step = make_case(tag, kind, spec)
    n = export_top(nm, step, tag)
    summary["cases"][tag] = {"csv": "M1_design_%s_top.csv" % tag,
                             "nodes": n, "load": str(spec)}
with open(os.path.join(OUT, "abaqus_design_summary.json"), "w") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
print("ABAQUS_M1_DESIGN_DONE")
