"""2.5 m primary-mirror FEA in Abaqus 2025 (whiffletree, plate level).

Purpose: produce *real solved* Abaqus surface data for the environment
sensitivity matrix S_env = [dz/deg_gravity, dz/K_thermal] used by
products/sensitivity.  This replaces the CalculiX plate model and the
synthetic CSVs for the gravity/thermal load cases.

Model (deliberately plate level, documented assumptions):
  * annular Zerodur disk, OD 2500 mm, ID 700 mm, thickness 260 mm, axis Z;
  * 36 bottom whiffletree pads in a 6-12-12-6 pattern
    (r = 375 / 625 / 875 / 1125 mm);
  * 24 side pads at mid-thickness (radial restraint);
  * pads modelled as springs to ground (k = 1e5 N/mm axial / 1e5 N/mm radial)
    instead of CAD contacts -> idealized whiffletree, NOT a real contact;
  * material Zerodur: E=90 GPa, nu=0.24, rho=2.53e-9 t/mm^3,
    alpha = 5e-8 /K (upper bound of 0 +/- 0.05 ppm/K);

Load cases:
  gravity_0deg    gravity along +X (optical axis horizontal)
  gravity_90deg   gravity along -Z (optical axis vertical)
  thermal_plus20K uniform +20 K
  thermal_minus20K uniform -20 K
  thermal_grad20K  radial soak gradient 0 K at ID -> +20 K at OD (assumed)

Output: data/fea_cases/abaqus_m1/<case>_top.csv
        columns x_mm,y_mm,z_mm,u1_mm,u2_mm,dz_nm  (same schema as the rest)

Run (Windows, Abaqus 2025):
  "D:\\SIMULIA\\Commands\\abq2025.bat" cae noGUI=scripts/abaqus_m1_whiffletree.py

Label: FEA-PROVISIONAL - real solver, idealized supports/thermal profile.
"""
from __future__ import print_function

import csv
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
    "TEL_ABAQUS_OUT",
    os.path.join(os.getcwd(), "data", "fea_cases", "abaqus_m1")))
SCRATCH = os.path.join(OUT, "odb")
for d in (OUT, SCRATCH):
    if not os.path.isdir(d):
        os.makedirs(d)
# Abaqus writes job files (.inp/.odb/.dat/.msg) into the process cwd.
os.chdir(SCRATCH)

R_OUT = 1250.0
R_IN = 350.0
THICK = 260.0
E = 90000.0            # MPa
NU = 0.24
RHO = 2.53e-9          # t/mm^3
ALPHA = 5.0e-8         # 1/K
K_BOTTOM = float(os.environ.get("TEL_ABAQUS_KB", 1.0e6))   # N/mm per pad, axial
K_SIDE = float(os.environ.get("TEL_ABAQUS_KS", 5.0e5))     # N/mm per pad, radial
SEED = float(os.environ.get("TEL_ABAQUS_SEED", 45.0))
SUFFIX = os.environ.get("TEL_ABAQUS_SUFFIX", "")
ONLY = [x for x in os.environ.get("TEL_ABAQUS_CASES", "").split(",") if x]
N_BANDS = 10
# A real whiffletree pad carries its load over a finite area.  Spreading the
# spring over a node patch (instead of one node) keeps the model mesh
# convergent: a single-node point spring diverges as the mesh is refined.
R_PAD = 70.0           # bottom pad patch radius, mm
R_PAD_SIDE = 45.0      # side pad patch radius, mm
BOTTOM_RINGS = ((375.0, 6), (625.0, 12), (875.0, 12), (1125.0, 6))
N_SIDE = 24

CASES = (
    ("gravity_0deg", "gravity", (9810.0, 0.0, 0.0)),
    ("gravity_90deg", "gravity", (0.0, 0.0, -9810.0)),
    ("thermal_plus20K", "thermal", 20.0),
    ("thermal_minus20K", "thermal", -20.0),
    ("thermal_grad20K", "thermal_gradient", 20.0),
)


def nearest_node(inst, x, y, z, radius=SEED):
    nodes = inst.nodes.getByBoundingSphere(center=(x, y, z), radius=radius)
    if not nodes:
        nodes = inst.nodes.getByBoundingSphere(center=(x, y, z), radius=radius * 3.0)
    if not nodes:
        raise RuntimeError("no mesh node near (%.1f, %.1f, %.1f)" % (x, y, z))
    best, bd = None, 1.0e30
    for n in nodes:
        c = n.coordinates
        d = math.sqrt((c[0] - x) ** 2 + (c[1] - y) ** 2 + (c[2] - z) ** 2)
        if d < bd:
            best, bd = n, d
    return best


def build_part(model):
    sketch = model.ConstrainedSketch(name="profile", sheetSize=3000.0)
    sketch.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R_OUT, 0.0))
    sketch.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R_IN, 0.0))
    part = model.Part(name="M1", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sketch, depth=THICK)
    mat = model.Material(name="Zerodur")
    mat.Elastic(table=((E, NU),))
    mat.Density(table=((RHO,),))
    mat.Expansion(table=((ALPHA,),))
    model.HomogeneousSolidSection(name="M1sec", material="Zerodur", thickness=None)
    part.SectionAssignment(region=(part.cells,), sectionName="M1sec")
    part.seedPart(size=SEED, deviationFactor=0.2, minSizeFactor=0.2)
    elem = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
    part.setElementType(regions=(part.cells,), elemTypes=(elem,))
    part.generateMesh()
    return part


def nodes_near(inst, cx, cy, cz, radius, ztol=2.0):
    """Node labels within `radius` of (cx, cy) in the plane z=cz."""
    labels = []
    for n in inst.nodes:
        c = n.coordinates
        if abs(c[2] - cz) > ztol:
            continue
        if (c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= radius * radius:
            labels.append(n.label)
    return labels


def nodes_ring(inst, r0, ang, cz, dr, ztol):
    """Nodes in a short arc at radius r0, around `ang`, near plane z=cz."""
    ca, sa = math.cos(ang), math.sin(ang)
    arc = dr / max(r0, 1.0) * 1.5      # angular half-width of the pad patch
    labels = []
    for n in inst.nodes:
        c = n.coordinates
        if abs(c[2] - cz) > ztol:
            continue
        r = math.sqrt(c[0] ** 2 + c[1] ** 2)
        if abs(r - r0) > dr:
            continue
        if abs(c[0] - r * ca) > arc * r0 + 1e-6:
            continue
        ang_n = math.atan2(c[1], c[0])
        d = math.atan2(math.sin(ang_n - ang), math.cos(ang_n - ang))
        if abs(d) <= arc:
            labels.append(n.label)
    return labels


def add_supports(model, asm, inst):
    """Distribute each whiffletree pad over a node patch (convergent model)."""
    ef = model.rootAssembly.engineeringFeatures
    n_spring = 0
    pad_count = 0
    for radius, count in BOTTOM_RINGS:
        for k in range(count):
            ang = 2.0 * math.pi * k / count
            name = "PAD_%02d" % pad_count
            pad_count += 1
            labels = nodes_near(inst, radius * math.cos(ang),
                                radius * math.sin(ang), 0.0, R_PAD,
                                ztol=1e-6)
            if not labels:
                raise RuntimeError("no nodes in pad patch %s" % name)
            asm.Set(name=name, nodes=inst.nodes.sequenceFromLabels(tuple(labels)))
            k_node = K_BOTTOM / float(len(labels))
            ef.SpringDashpotToGround(
                name=name + "_D3", region=asm.sets[name], orientation=None,
                dof=3, springBehavior=ON, springStiffness=k_node)
            n_spring += 1
    for k in range(N_SIDE):
        ang = 2.0 * math.pi * k / N_SIDE
        name = "SIDE_%02d" % k
        labels = []
        for ztol in (2.0, SEED, 2.0 * SEED):
            labels = nodes_ring(inst, R_OUT, ang, THICK / 2.0, R_PAD_SIDE,
                                ztol)
            if labels:
                break
        if not labels:
            raise RuntimeError("no nodes in side pad %s" % name)
        asm.Set(name=name, nodes=inst.nodes.sequenceFromLabels(tuple(labels)))
        k_node = K_SIDE / float(len(labels))
        for dof in (1, 2):
            ef.SpringDashpotToGround(
                name="%s_D%d" % (name, dof), region=asm.sets[name],
                orientation=None, dof=dof, springBehavior=ON,
                springStiffness=k_node)
            n_spring += 1
    return pad_count, n_spring


def add_load(model, asm, inst, kind, spec):
    if kind == "gravity":
        model.StaticStep(name="load", previous="Initial", timePeriod=1.0,
                         nlgeom=OFF)
        model.Gravity(name="g", createStepName="load",
                      comp1=spec[0], comp2=spec[1], comp3=spec[2])
        return "load"
    model.StaticStep(name="load", previous="Initial", timePeriod=1.0,
                     nlgeom=OFF)
    if kind == "thermal":
        model.Temperature(name="dT", createStepName="load",
                          region=asm.sets["ALL_NODES"],
                          distributionType=UNIFORM,
                          crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                          magnitudes=(spec,))
        return "load"
    # radial soak gradient: 0 K at ID -> +20 K at OD, 10 annular bands
    bands = {}
    for n in inst.nodes:
        c = n.coordinates
        r = math.sqrt(c[0] ** 2 + c[1] ** 2)
        frac = min(max((r - R_IN) / (R_OUT - R_IN), 0.0), 1.0)
        b = min(int(frac * N_BANDS), N_BANDS - 1)
        bands.setdefault(b, []).append(n.label)
    for b in sorted(bands):
        nm = "BAND_%02d" % b
        asm.Set(name=nm, nodes=inst.nodes.sequenceFromLabels(tuple(bands[b])))
        frac = (b + 0.5) / N_BANDS
        model.Temperature(name="dT_" + nm, createStepName="load",
                          region=asm.sets[nm],
                          distributionType=UNIFORM,
                          crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                          magnitudes=(spec * frac,))
    return "load"


def make_case(tag, kind, spec):
    name = "M1_" + tag + SUFFIX
    if name in mdb.models:
        del mdb.models[name]
    model = mdb.Model(name=name)
    part = build_part(model)
    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    inst = asm.Instance(name="M1-1", part=part, dependent=ON)
    nb, ns = add_supports(model, asm, inst)
    asm.Set(name="ALL_NODES", nodes=inst.nodes)
    step = add_load(model, asm, inst, kind, spec)
    model.fieldOutputRequests["F-Output-1"].setValues(variables=("U", "RF"))
    job = mdb.Job(name=name, model=name, type=ANALYSIS, numCpus=1,
                  explicitPrecision=SINGLE)
    print("solving %s (pads bottom=%d side=%d)" % (name, nb, ns))
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    return name, step


def export_top(name, step):
    odb_path = os.path.join(SCRATCH, name + ".odb")
    if not os.path.exists(odb_path):
        raise RuntimeError("missing ODB " + odb_path)
    odb = openOdb(odb_path, readOnly=True)
    inst = odb.rootAssembly.instances["M1-1"]
    coords = {}
    for n in inst.nodes:
        coords[n.label] = n.coordinates
    frame = odb.steps[step].frames[-1]
    u = frame.fieldOutputs["U"]
    rows, sum_rf = [], 0.0
    for v in u.values:
        c = coords.get(v.nodeLabel)
        if c is None:
            continue
        if c[2] >= THICK - 1.0:
            rows.append((c[0], c[1], c[2], v.data[0], v.data[1], v.data[2]))
    out_csv = os.path.join(OUT, name + "_top.csv")
    with open(out_csv, "w") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x_mm", "y_mm", "z_mm", "u1_mm", "u2_mm", "dz_nm"])
        for x, y, z, u1, u2, u3 in rows:
            wr.writerow(["%.6f" % x, "%.6f" % y, "%.6f" % z,
                         "%.9f" % u1, "%.9f" % u2, "%.6f" % (u3 * 1.0e6)])
    # reaction check: sum(RF3) must balance the weight for gravity cases
    rf_txt = ""
    try:
        rf = frame.fieldOutputs["RF"]
        rf = frame.fieldOutputs.get("RF")
        total = 0.0
        for v in rf.values:
            total += v.data[2]
        rf_txt = " sum_RF3=%.1f N" % total
    except Exception:
        rf_txt = " (no RF output requested)"
    odb.close()
    print("  exported %d top nodes -> %s%s" % (len(rows), out_csv, rf_txt))
    return len(rows)


print("OUT =", OUT, "| seed =", SEED, "| k_bottom =", K_BOTTOM,
      "| k_side =", K_SIDE, "| suffix =", SUFFIX)
for tag, kind, spec in CASES:
    if ONLY and tag not in ONLY:
        continue
    nm, step = make_case(tag, kind, spec)
    export_top(nm, step)
print("ABAQUS_M1_DONE")
